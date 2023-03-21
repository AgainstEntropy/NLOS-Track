import os
import time

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

import models


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = -1
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, optimizer, scheduler, save_dir, acc=00):
    model_paras = model.checkpoint()
    optim_paras = optimizer.checkpoint()
    scheduler_main_paras = scheduler.checkpoint()

    save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = os.path.join(save_dir, f'{acc:.1f}_{save_time}.pt')
    torch.save({
        "model_paras": model_paras,
        "optim_paras": optim_paras,
        "scheduler_paras": scheduler_main_paras
    }, save_path)

    print(f"\nSuccessfully saved model, optimizer and scheduler to {save_path}")


def get_device(model):
    if next(model.parameters()).device.type == 'cuda':
        index = next(model.parameters()).device.index
        device = torch.device(f'cuda:{index}')
    else:
        device = torch.device('cpu')
    return device


def load_model(run_name: str,
               log_dir: str,
               ckpt_name: str = 'best') -> torch.nn.Module:
    run_dir = os.path.join(log_dir, run_name)
    checkpoint = torch.load(os.path.join(run_dir, f'checkpoints/{ckpt_name}.pth'))
    with open(os.path.join(run_dir, 'configs.yaml'), 'r') as stream:
        run_config = yaml.load(stream, Loader=yaml.FullLoader)

    model_dict = {
        'PAC_Net': models.PAC_Net,
        'P_Net': models.P_Net,
        'C_Net': models.C_Net,
        'baseline': models.NLOS_baseline
    }

    model_name = run_config['model_configs'].pop('model_name')
    print('Min val loss is:', checkpoint['min_loss_total'])
    model_builder = model_dict[model_name]
    model = model_builder(**run_config['model_configs'])

    load_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(load_state_dict)
    print('Successfully load model parameters!')

    return model.eval()


def fig2array(fig: plt.Figure):
    from PIL import Image

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    image_array = Image.frombytes("RGBA", (w, h), buf.tostring())
    image_array = np.asarray(image_array)
    plt.close(fig)
    return image_array
