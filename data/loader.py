import io
import os
from typing import Tuple, Union

import numpy as np
import mmcv
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize


def load_frames(
        root: str,
        frame_range: Union[None, Tuple[int, int]] = None,
        output_size: Union[None, Tuple[int, int]] = None,
        rgb_only=True
) -> torch.Tensor:
    frame_list = sorted([f for f in os.listdir(root) if f.endswith('.png')])
    if frame_range is not None:
        frame_list = frame_list[frame_range[0]: frame_range[1]]
    frame_paths = [os.path.join(root, f) for f in frame_list]

    C, H, W = read_image(frame_paths[0]).shape
    frame_num = len(frame_list)
    if C == 4 and rgb_only:
        frames = torch.zeros((frame_num, 3, H, W))
    else:
        frames = torch.zeros((frame_num, C, H, W))
    for i in range(frame_num):
        frame = read_image(frame_paths[i])  # (C, H, W)
        if C == 4 and rgb_only:
            frame = frame[:3]
        frames[i] = frame
    if output_size is not None:
        frames = resize(frames, size=output_size)
    return frames  # (T, C, H, W)


class npy_loader(object):
    def __init__(self):
        self.file_client = mmcv.fileio.FileClient(backend='petrel')

    def get_item(self, file_path: str):
        npy_buffer = self.file_client.get(file_path)
        # return np.frombuffer(npy_buffer)
        with io.BytesIO(npy_buffer) as f:
            return np.load(f)  # , encoding='bytes', allow_pickle=True
