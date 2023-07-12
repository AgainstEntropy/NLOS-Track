import argparse
import os

import torch
import torch.multiprocessing as mp
import yaml


def main(cfg):
    dist_cfgs = cfg['distributed_configs']

    os.makedirs(cfg['log_configs']['log_dir'], exist_ok=True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = dist_cfgs['device_ids']
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    world_size = len(dist_cfgs['device_ids'].split(','))
    dist_cfgs['distributed'] = True if world_size > 1 else False
    dist_cfgs['world_size'] = world_size
    cfg['loader_kwargs']['batch_size'] = cfg['train_configs']['batch_size'] // world_size

    print("Allocating workers...")
    if dist_cfgs['distributed']:
        mp.spawn(worker, nprocs=world_size, args=(cfg,))
    else:
        worker(0, cfg)


def worker(rank, cfg):
    torch.cuda.set_device(rank)
    cfg['distributed_configs']['local_rank'] = rank

    from utils import Trainer_tracking
    trainer = Trainer_tracking(cfg)

    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_file', type=str, default='default')

    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--rnn_hdim', type=int, default=128)

    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--v_loss_alpha', type=float, default=500)
    parser.add_argument('--loss_total_alpha', type=float, default=1000)
    parser.add_argument('-r', '--resume', action='store_true', help='load previously saved checkpoint')

    parser.add_argument('-lr_b', '--lr_backbone', type=float, default=3e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=2.0e-3)

    parser.add_argument('-T', '--cos_T', type=int, default=70)

    parser.add_argument('-g', '--gpu_ids', type=lambda x: x.replace(" ", ""), default='0',
                        help='available gpu ids')
    parser.add_argument('--port', type=str, default='6666', help='port number of distributed init')

    args = parser.parse_args()

    config_file = os.path.join('configs', f'{args.cfg_file}.yaml')
    print(f'Reading config file: {config_file}')
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    config['model_configs']['warm_up'] = args.warm_up
    config['model_configs']['pretrained'] = args.pretrained
    config['model_configs']['rnn_hdim'] = args.rnn_hdim
    config['model_configs']['model_name'] = args.model_name

    config['dataset_configs']['route_len'] += args.warm_up

    config['train_configs']['batch_size'] = args.batch_size
    config['train_configs']['v_loss_alpha'] = args.v_loss_alpha
    config['train_configs']['loss_total_alpha'] = args.loss_total_alpha
    config['train_configs']['resume'] = args.resume

    config['optim_kwargs']['lr'] = args.lr_backbone
    config['optim_kwargs']['weight_decay'] = args.weight_decay

    config['schedule_configs']['cos_T'] = args.cos_T

    config['distributed_configs']['device_ids'] = args.gpu_ids
    config['distributed_configs']['port'] = args.port

    main(config)
