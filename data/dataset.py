import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from scipy.io import loadmat
from tqdm import tqdm

from .loader import load_frames, npy_loader


class TrackingDataset(Dataset):
    def __init__(self,
                 dataset_root: str,
                 noise_factor: float = 0,
                 datalist_file: str = 'dataset_list.txt',
                 route_len: int = 128,
                 total_len: int = 250) -> None:
        self.datafile_name = f'video_N{noise_factor}.npy'
        self.total_len = total_len
        self.num_frames = route_len

        abs_file_path = os.path.join(dataset_root, datalist_file)
        assert os.path.exists(abs_file_path)

        self.dataset_root = dataset_root
        with open(abs_file_path, mode='r') as f:
            self.png_dirs = sorted([line.strip() for line in f.readlines()])

        self.npy_loader = npy_loader()

    def __len__(self):
        return len(self.png_dirs)

    def __getitem__(self, idx):
        abs_png_dir = os.path.join(self.dataset_root, self.png_dirs[idx])
        npy_file = os.path.join(abs_png_dir, self.datafile_name)
        npy = self.npy_loader.get_item(npy_file)

        start_frame = random.randint(0, self.total_len - self.num_frames)
        npy = npy[:, start_frame:start_frame + self.num_frames]  # (3, T, H, W) or (3, T-1, H, W)

        mat_file = loadmat(os.path.join(abs_png_dir, 'route.mat'))
        routes = mat_file['route'][start_frame:start_frame + self.num_frames]  # (T, 2)
        map_size = mat_file['map_size']  # (1, 2)

        return torch.from_numpy(npy), torch.from_numpy(routes).float(), torch.from_numpy(map_size).float()


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


class raw_png_processor(object):
    def __init__(self, dataset_root: str):

        self.target_png_num = None
        self.actions_num = None
        self.data_dict = None
        self.actions = None
        self.raw_data_root = dataset_root
        assert os.path.exists(self.raw_data_root)
        self.render_result_file = os.path.join(self.raw_data_root, 'render_state.txt')
        self.ds_list_file = os.path.join(self.raw_data_root, 'dataset_list.txt')

    def check_render_data(self, target_png_num: int, resume=False):
        self.target_png_num = target_png_num
        if resume:
            with open(self.render_result_file, mode='r') as f:
                completed_dirs = [line.strip() for line in f.readlines()]
        else:
            completed_dirs = []

        png_dirs = os.listdir(self.raw_data_root)
        for png_dir in tqdm(png_dirs):
            if png_dir in completed_dirs:
                continue
            abs_dir = os.path.join(self.raw_data_root, png_dir)
            if not os.path.isdir(abs_dir):
                continue
            png_num = len([f for f in os.listdir(abs_dir) if f.endswith('.png')])
            if png_num == target_png_num:
                completed_dirs.append(png_dir)

        with open(self.render_result_file, mode='a+') as f:
            f.truncate(0)
            f.writelines(sorted([line + '\n' for line in completed_dirs]))

    def build_dataset(self,
                            noise_factor: float = 0,
                            resize: tuple[int, int] = (128, 128)):

        npy_name = f'video_{resize[0]}'

        with open(self.render_result_file, mode='r') as f:
            png_dirs = [line.strip() for line in f.readlines()]

        for png_dir in tqdm(png_dirs):
            png_abs_dir = os.path.join(self.raw_data_root, png_dir)
            npy_path = os.path.join(png_abs_dir, f'{npy_name}_N{noise_factor}.npy')
            if os.path.exists(npy_path):
                continue
            frames = load_frames(png_abs_dir, output_size=resize)  # (T, 3, H, W)
            if noise_factor != 0:
                frames += 255 * noise_factor * torch.randn_like(frames)
            save_array = frames.permute(1, 0, 2, 3).numpy().astype(np.int8)  # (3, T, H, W)
            assert save_array.shape == (3, self.target_png_num, 128, 128)

            np.save(npy_path, save_array)

        self.build_dataset_list()

    def build_dataset_list(self):
        file_name = 'video_N0.npy'

        with open(self.render_result_file, mode='r') as f:
            png_dirs = [line.strip() for line in f.readlines()]

        lines = []
        for nm_dir in tqdm(sorted(png_dirs)):
            abs_dir = os.path.join(self.raw_data_root, nm_dir)
            abs_file_path = os.path.join(abs_dir, file_name)
            mat_path = os.path.join(abs_dir, 'route.mat')
            if os.path.exists(abs_file_path) and os.path.exists(mat_path):
                lines.append(f'{nm_dir}\n')

        with open(self.ds_list_file, mode='a+') as f:
            f.truncate(0)
            f.writelines(lines)

        print(f'Successfully write into {self.ds_list_file}!')
