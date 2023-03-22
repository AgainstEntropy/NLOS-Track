import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from scipy.io import loadmat

from .loader import npy_loader


class TrackingDataset(Dataset):
    def __init__(self,
                 dataset_root: str,
                 data_type: str = 'render',
                 route_len: int = 128,
                 use_fileclient: bool = False) -> None:
        self.dataset_root = dataset_root
        self.total_len = 320 if data_type == 'render' else 250
        self.num_frames = route_len
        self.npy_name = 'video.npy'

        dataset_dir = os.path.join(dataset_root, data_type)
        self.dirs = sorted(os.listdir(dataset_dir))

        if use_fileclient:
            self.npy_loader = npy_loader()
            self.load_npy = self.npy_loader.get_item
        else:
            self.load_npy = np.load

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        abs_png_dir = os.path.join(self.dataset_root, self.dirs[idx])
        npy_file = os.path.join(abs_png_dir, self.npy_name)
        video = self.load_npy(npy_file)

        start_frame = random.randint(0, self.total_len - self.num_frames)
        video = video[:, start_frame:start_frame + self.num_frames]  # (3, T, H, W) or (3, T-1, H, W)

        mat_file = loadmat(os.path.join(abs_png_dir, 'route.mat'))
        routes = mat_file['route'][start_frame:start_frame + self.num_frames]  # (T, 2)
        map_size = mat_file['map_size']  # (1, 2)

        return torch.from_numpy(video), torch.from_numpy(routes).float(), torch.from_numpy(map_size).float()


def split_dataset(
        dataset_root: str,
        noise_factor: float = 0,
        route_len: int = 128,
        total_len: int = 250,
        phase: str = 'train',
        train_ratio: float = 0.8,
        **kwargs
):
    full_dataset = TrackingDataset(dataset_root=dataset_root, noise_factor=noise_factor,
                                   route_len=route_len, total_len=total_len)

    if phase == 'train':
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, val_size])
    elif phase == 'test':
        return full_dataset
