from torch import Tensor
from torchvision.transforms.functional import resize


def sub_mean(frames: Tensor) -> Tensor:
    mean_frame = frames.mean(axis=0, keepdim=True)
    frames_sub_mean = frames.sub(mean_frame)

    return frames_sub_mean


def diff(frames: Tensor) -> Tensor:
    return frames[1:].sub(frames[:-1])


def normalize(frame: Tensor):
    return (frame - frame.min()) / (frame.max() - frame.min())


def resize_video(frames: Tensor, bias_ratio: float = None, output_size: tuple = (128, 128)) -> Tensor:
    T, C, H, W = frames.shape
    crop_idx = (W - H) // 2
    if bias_ratio is not None:
        crop_idx -= int(W * bias_ratio)
    output_frames = frames[:, :, :, crop_idx:crop_idx + H]
    if output_size is not None:
        output_frames = resize(output_frames, size=output_size)

    return output_frames
