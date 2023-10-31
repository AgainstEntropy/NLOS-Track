import os
import platform
import random
import time
from decimal import Decimal

import numpy as np
import wandb
import yaml
from loguru import logger
from prettytable import PrettyTable
from tqdm import tqdm
from fairscale.optim.oss import OSS
import torch
from torch import optim, nn, distributed
from torch.cuda.amp import GradScaler, autocast
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

import models
from data.dataset import split_dataset
from .vis import draw_routes
from .tools import AverageMeter, load_model
from .metric import compute_batch_metrics


# cudnn.benchmark = True


def seed_worker(worker_id):
    # print(torch.initial_seed())
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _set_seed(seed, deterministic=False):
    """
    seed manually to make runs reproducible
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option
        for CUDNN backend
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


class Trainer_Base(object):
    def __init__(self, cfg):
        tic = time.time()
        self.dist_cfgs = cfg['distributed_configs']
        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Loading configurations...")
        self.cfg = cfg
        self.model_cfgs = cfg['model_configs']
        self.train_cfgs = cfg['train_configs']
        self.dataset_cfgs = cfg['dataset_configs']
        self.loader_kwargs = cfg['loader_kwargs']
        self.optim_kwargs = cfg['optim_kwargs']
        self.schedule_cfgs = cfg['schedule_configs']
        self.log_cfgs = cfg['log_configs']

        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Initializing trainer...")
        if self.dist_cfgs['distributed']:
            distributed.init_process_group(backend='nccl',
                                           init_method='tcp://127.0.0.1:' + self.dist_cfgs['port'],
                                           world_size=self.dist_cfgs['world_size'],
                                           rank=self.dist_cfgs['local_rank'])
        _set_seed(self.train_cfgs['seed'] + self.dist_cfgs['local_rank'], deterministic=True)
        if torch.cuda.is_available():
            self.device = f'cuda:{self.dist_cfgs["local_rank"]}'
        else:
            self.device = "cpu"
        self.dist_cfgs['device'] = self.device

        self.save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        if self.log_cfgs['log_dir'].startswith('/'):
            log_root = self.log_cfgs['log_dir']
        else:
            log_root = os.path.join(os.getcwd(), self.log_cfgs['log_dir'])
        self.log_dir = os.path.join(log_root, self.save_time)
        self.ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.dist_cfgs['local_rank'] == 0:
            with open(os.path.join(self.log_dir, 'configs.yaml'), 'w', encoding="utf-8") as f:
                yaml.safe_dump(self.cfg, f, default_flow_style=False, allow_unicode=True)

        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Loading dataset...")
        (self.train_loader, self.train_sampler), (self.val_loader, self.val_sampler) = self._load_dataset()

        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Building model...")
        self.model_name = self._build_model()
        if self.dist_cfgs['distributed']:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model,
                             device_ids=[self.dist_cfgs['local_rank']],
                             output_device=self.dist_cfgs['local_rank'],
                             find_unused_parameters=True)

        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Loading optimizer...")
        self._load_optimizer()

        if self.dist_cfgs['local_rank'] == 0:
            print(f"{time.time() - tic:.2f} sec are used to initialize a Trainer.")

        self.start_epoch = 0
        self.steps = 0
        self.epoch = 0
        self.train_min_loss = float('inf')

    def _load_dataset(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _load_optimizer(self):
        base_optimizer = None
        optim_type = self.optim_kwargs.pop('optimizer')
        if optim_type == 'SGD':
            base_optimizer = optim.SGD
            self.optim_kwargs['momentum'] = 0.9
        elif optim_type == 'Adam':
            base_optimizer = optim.Adam
            # self.optim_kwargs['betas'] = (0.9, 0.999)
        elif optim_type == 'AdamW':
            base_optimizer = optim.AdamW
            # self.optim_kwargs['betas'] = (0.9, 0.999)
        else:
            print(f"{optim_type} not support.")
            exit(0)

        if self.dist_cfgs['distributed']:
            # Wrap a base optimizer into OSS
            self.optimizer = OSS(
                optim=base_optimizer,
                params=self.model.parameters(),
                **self.optim_kwargs,
            )
        else:
            self.optimizer = base_optimizer(
                params=self.model.parameters(),
                **self.optim_kwargs,
            )

        if self.schedule_cfgs['schedule_type'] == 'cosine_warm':
            self.schedule_cfgs['max_epoch'] = \
                int((self.schedule_cfgs['cos_mul'] ** self.schedule_cfgs['cos_iters'] - 1) / \
                    (self.schedule_cfgs['cos_mul'] - 1) * self.schedule_cfgs['cos_T'])
            self.scheduler = \
                optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                               T_0=self.schedule_cfgs['cos_T'],
                                                               T_mult=self.schedule_cfgs['cos_mul'])
        elif self.schedule_cfgs['schedule_type'] == 'cosine':
            self.schedule_cfgs['max_epoch'] = self.schedule_cfgs['cos_T']
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.schedule_cfgs['cos_T'])

        if self.train_cfgs['amp']:
            self.scaler = GradScaler()

        self.optim_kwargs['optimizer'] = optim_type

    def _init_recorder(self, log_train_cfg):
        wandb.init(project=self.train_cfgs['project_name'],
                   name=self.save_time, dir=self.log_dir, config=log_train_cfg,
                   settings=wandb.Settings(start_method="fork"))
        wandb.watch(self.model)

        config_table = PrettyTable()
        config_table.add_column('Phase', list(log_train_cfg))
        config_table.add_column('Value', list(log_train_cfg.values()))
        logger.info('\n' + config_table.get_string())

    def load_checkpoint(self, path):
        raise NotImplementedError


class Trainer_tracking(Trainer_Base):
    def __init__(self, cfg):
        super(Trainer_tracking, self).__init__(cfg=cfg)

        if self.dist_cfgs['local_rank'] == 0:
            log_train_cfg = {
                "model_name": self.model_name,
                **self.model_cfgs,
                "batch_size": self.train_cfgs['batch_size'],
                "v_loss_alpha": self.train_cfgs['v_loss_alpha'],
                "loss_total_alpha": self.train_cfgs['loss_total_alpha'],
                "resume": self.train_cfgs['resume'],
                "route_len": self.dataset_cfgs['route_len'],
                "noise_factor": self.dataset_cfgs['noise_factor'],
                **self.optim_kwargs,
                "epochs": self.schedule_cfgs['max_epoch'],
            }

            self._init_recorder(log_train_cfg)

        self.val_metrics = {'x_loss': 0.0,
                            'v_loss': 0.0,
                            'min_loss_total': float('inf'),
                            'pcm': float('inf'),
                            'area': float('inf'),
                            'dtw': float('inf'),
                            'best_epoch': 0}

    def _load_dataset(self):
        train_dataset, val_dataset = split_dataset(**self.dataset_cfgs)

        if self.dist_cfgs['distributed']:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=True)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(train_dataset, **self.loader_kwargs, worker_init_fn=seed_worker, drop_last=True)
        val_loader = DataLoader(val_dataset, **self.loader_kwargs, worker_init_fn=seed_worker, drop_last=False)
        return (train_loader, train_sampler), (val_loader, val_sampler)

    def _build_model(self):
        model_name = self.model_cfgs.pop('model_name')
        model_builder = {
            'PAC_Net': models.PAC_Net,
            'P_Net': models.P_Net,
            'C_Net': models.C_Net,
            'baseline': models.NLOS_baseline,
        }[model_name]

        self.model = model_builder(**self.model_cfgs)
        if self.train_cfgs['resume']:
            checkpoint_path = self.train_cfgs['resume_path']
            self.load_checkpoint(checkpoint_path)
        self.model.to(self.device)

        return model_name

    def run(self):
        if self.dist_cfgs['local_rank'] == 0:
            logger.info("--- Begin to run! ---")
        for epoch in range(self.start_epoch, self.schedule_cfgs['max_epoch']):

            if self.dist_cfgs['distributed']:
                self.train_sampler.set_epoch(epoch)

            train_loss, train_metric = self.train(epoch)
            val_loss, val_metric = self.val(epoch)
            self.epoch += 1  # (1->70)

            if self.dist_cfgs['local_rank'] == 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    wandb.log({f"optimizer/lr_group_{i}": param_group['lr']}, step=epoch + 1)
                wandb.log({
                    'Loss/train/loss_x': train_loss[0],
                    'Loss/train/loss_v': train_loss[1],
                    'Loss/val/loss_x': val_loss[0],
                    'Loss/val/loss_v': val_loss[1],
                    'Loss/val/min_loss_total': self.val_metrics['min_loss_total'],
                }, step=epoch + 1)
                wandb.log({
                    'Metric/train/pcm': train_metric[0],
                    'Metric/train/area': train_metric[1],
                    'Metric/train/dtw': train_metric[2],
                    'Metric/val/pcm': val_metric[0],
                    'Metric/val/area': val_metric[1],
                    'Metric/val/dtw': val_metric[2],
                }, step=epoch + 1)
                if self.epoch % 5 == 0:
                    logger.info('Logging images...')
                    self.test_plot(epoch=self.epoch, phase='train')
                    self.test_plot(epoch=self.epoch, phase='val')

            self.scheduler.step()

            if ((epoch + 1) % self.log_cfgs['save_epoch_interval'] == 0) \
                        or (epoch + 1) == self.schedule_cfgs['max_epoch']:
                checkpoint_path = os.path.join(self.ckpt_dir, f"epoch_{(epoch + 1)}.pth")
                self.save_checkpoint(checkpoint_path)

        if self.dist_cfgs['local_rank'] == 0:
            wandb.finish()

        if self.dist_cfgs['distributed']:
            distributed.destroy_process_group()

    def train(self, epoch):
        self.model.train()
        len_loader = len(self.train_loader)
        iter_loader = iter(self.train_loader)

        # loss
        x_loss_recorder = AverageMeter()
        v_loss_recorder = AverageMeter()

        # metric
        pcm_recorder = AverageMeter()
        area_recorder = AverageMeter()
        dtw_recorder = AverageMeter()

        pbar = None
        if self.dist_cfgs['local_rank'] == 0:
            pbar = tqdm(total=len_loader,
                        dynamic_ncols=True,
                        ascii=(platform.version() == 'Windows'))

        for _ in range(len_loader):
            try:
                inputs, labels, map_sizes = next(iter_loader)
            except Exception as e:
                logger.critical(e)
                continue

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            map_sizes = map_sizes.to(self.device)

            batch_size = inputs.size(0)

            if self.train_cfgs['amp']:
                with autocast():
                    (x_loss, v_loss), preds = self.model((inputs, labels))
                loss = (x_loss + v_loss * self.train_cfgs['v_loss_alpha']) * self.train_cfgs['loss_total_alpha']
                self.scaler.scale(loss).backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                (x_loss, v_loss), preds = self.model((inputs, labels))
                loss = (x_loss + v_loss * self.train_cfgs['v_loss_alpha']) * self.train_cfgs['loss_total_alpha']
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1
            x_loss = x_loss.detach().clone()
            v_loss = v_loss.detach().clone()
            if self.dist_cfgs['distributed']:
                distributed.reduce(x_loss, 0)
                x_loss /= self.dist_cfgs['world_size']
                v_loss /= self.dist_cfgs['world_size']
            x_loss_recorder.update(x_loss.item(), batch_size)
            v_loss_recorder.update(v_loss.item(), batch_size)

            T = preds.shape[1]
            preds = preds * map_sizes
            labels = labels * map_sizes
            pcm, area, dtw = compute_batch_metrics(preds, labels[:, -T:])
            pcm_recorder.update(pcm.item(), batch_size)
            area_recorder.update(area.item(), batch_size)
            dtw_recorder.update(dtw.item(), batch_size)

            if self.dist_cfgs['local_rank'] == 0:
                last_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
                last_lr_string = "lr " + ' '.join(f"{Decimal(lr):.2E}" for lr in last_lr)

                pbar.set_description(
                    f"train epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
                    # f"Iter {self.steps}/{len_loader * self.schedule_cfgs['max_epoch']}  "
                    f"{last_lr_string}  "
                    f"----  "
                    f"x_loss: {x_loss_recorder.avg:.3E}  "
                    f"v_loss: {v_loss_recorder.avg:.3E}  "
                    f"----  "
                    f"area: {area_recorder.avg:.4f}  "
                    f"dtw: {dtw_recorder.avg:.3f}  "
                    f"pcm: {pcm_recorder.avg:.3f}  "
                )
                pbar.update()

                if self.steps % self.log_cfgs['snapshot_interval'] == 0:
                    checkpoint_path = os.path.join(self.ckpt_dir, "latest.pth")
                    self.save_checkpoint(checkpoint_path)

        if self.dist_cfgs['local_rank'] == 0:
            pbar.close()

        return (x_loss_recorder.avg, v_loss_recorder.avg), \
                   (pcm_recorder.avg, area_recorder.avg, dtw_recorder.avg)

    def val(self, epoch):
        self.model.eval()
        len_loader = len(self.val_loader)
        iter_loader = iter(self.val_loader)

        # loss
        x_loss_recorder = AverageMeter()
        v_loss_recorder = AverageMeter()

        # metric
        pcm_recorder = AverageMeter()
        area_recorder = AverageMeter()
        dtw_recorder = AverageMeter()

        pbar = None
        if self.dist_cfgs['local_rank'] == 0:
            pbar = tqdm(total=len_loader,
                        dynamic_ncols=True,
                        ascii=(platform.version() == 'Windows'))

        for step in range(len_loader):
            try:
                inputs, labels, map_sizes = next(iter_loader)
            except Exception as e:
                logger.critical(e)
                continue

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            map_sizes = map_sizes.to(self.device)

            batch_size = inputs.size(0)

            with torch.no_grad():
                if self.train_cfgs['amp']:
                    with autocast():
                        (x_loss, v_loss), preds = self.model((inputs, labels))
                else:
                    (x_loss, v_loss), preds = self.model((inputs, labels))

            x_loss = x_loss.detach().clone()
            v_loss = v_loss.detach().clone()
            if self.dist_cfgs['distributed']:
                distributed.reduce(x_loss, 0)
                x_loss /= self.dist_cfgs['world_size']
                v_loss /= self.dist_cfgs['world_size']
            x_loss_recorder.update(x_loss.item(), batch_size)
            v_loss_recorder.update(v_loss.item(), batch_size)

            T = preds.shape[1]
            preds = preds * map_sizes
            labels = labels * map_sizes

            pcm, area, dtw = compute_batch_metrics(preds, labels[:, -T:])
            pcm_recorder.update(pcm.item(), batch_size)
            area_recorder.update(area.item(), batch_size)
            dtw_recorder.update(dtw.item(), batch_size)

            if self.dist_cfgs['local_rank'] == 0:
                pbar.set_description(
                    f"val epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
                    f"Step {step}/{len_loader}  "
                    f"------  "
                    f"x_loss: {x_loss_recorder.avg:.3E}  "
                    f"v_loss: {v_loss_recorder.avg:.3E}  "
                    f"----  "
                    f"area: {area_recorder.avg:.4f}  "
                    f"dtw: {dtw_recorder.avg:.4f}  "
                    f"pcm: {pcm_recorder.avg:.4f}  ")
                pbar.update()

        if self.dist_cfgs['local_rank'] == 0:
            pbar.close()

            self.val_metrics['x_loss'] = x_loss_recorder.avg
            self.val_metrics['v_loss'] = v_loss_recorder.avg
            loss_total = x_loss_recorder.avg + v_loss_recorder.avg * self.train_cfgs['v_loss_alpha']
            if loss_total < self.val_metrics['min_loss_total']:
                self.val_metrics['min_loss_total'] = loss_total
                self.val_metrics['best_epoch'] = epoch + 1

                checkpoint_path = os.path.join(self.ckpt_dir, "best.pth")
                self.save_checkpoint(checkpoint_path)

            self.val_metrics['pcm'] = pcm_recorder.avg
            self.val_metrics['area'] = area_recorder.avg
            self.val_metrics['dtw'] = dtw_recorder.avg

            names = ['x_loss', 'v_loss', 'pcm', 'area', 'dtw', 'min_loss_total', 'best_epoch']
            res_table = PrettyTable(names)
            metrics = [self.val_metrics[name] for name in names]
            res_table.add_row([f"{m:.4}" if type(m) is float else m for m in metrics[:-1]] + [metrics[-1]])

            logger.info(
                f'Performance on validation set at epoch: {epoch + 1}\n{res_table.get_string()}'
            )

        return (self.val_metrics['x_loss'], self.val_metrics['v_loss']), \
                   (self.val_metrics['pcm'], self.val_metrics['area'], self.val_metrics['dtw'])

    def test_plot(self, epoch, phase: str):
        assert phase in {'train', 'val'}
        self.model.eval()
        iter_loader = iter(self.val_loader) if phase == 'val' else iter(self.train_loader)
        frames, gt_routes, map_sizes = next(iter_loader)
        frames = frames[:6].to(self.device)
        gt_routes = gt_routes[:6].to(self.device)
        map_sizes = map_sizes[:6].to(self.device)
        with torch.no_grad():
            if self.train_cfgs['amp']:
                with autocast():
                    loss, pred_routes = self.model((frames, gt_routes))
            else:
                loss, pred_routes = self.model((frames, gt_routes))

        for idx, (gt, pred, map_size) in enumerate(zip(
                gt_routes.cpu().numpy(), pred_routes.cpu().numpy(), map_sizes.cpu().numpy())):
            mark_T = None if 'warmup' not in self.model_name else self.model_cfgs['warm_up']
            fig = draw_routes(routes=(gt, pred), return_mode='fig_array', mark_T=mark_T)
            # pcm, area, dtw = compute_track_metrics(gt, pred)
            wandb.log({f'{phase} route:{idx}': wandb.Image(fig, caption=f"map_size: {map_size}")},
                      step=epoch)

    def save_checkpoint(self, path):
        # self.optimizer.consolidate_state_dict()
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])

        if self.dist_cfgs['local_rank'] == 0:
            save_dict = {
                'model': self.model.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'iteration': self.steps,
                **self.val_metrics
            }
            torch.save(save_dict, path)

    def load_checkpoint(self, path):
        # ckpt = None
        # if self.dist_cfgs['local_rank'] == 0:
        #     ckpt = torch.load(path, map_location={'cuda:0': f'cuda:{self.dist_cfgs["local_rank"]}'})
        # self.model.load_state_dict(ckpt['model'])
        # self.optimizer.load_state_dict(ckpt['optimizer'])
        # self.start_epoch = ckpt['epoch']
        # self.steps = ckpt['iteration']
        # self.val_metrics['best_epoch'] = ckpt['best_epoch']
        # self.val_metrics['min_loss'] = ckpt['min_val_loss']
        self.model = load_model(run_name=path,
                                log_dir=self.log_cfgs['log_dir'])
