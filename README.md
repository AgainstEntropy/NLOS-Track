# NLOS Tracking

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://againstentropy.github.io/NLOS-Track/)
[![arXiv](https://img.shields.io/badge/arXiv-2303.11791-b31b1b.svg)](https://arxiv.org/abs/2303.11791)

Official codes of CVPR 2023 [Paper](https://arxiv.org/abs/2303.11791) | _Propagate And Calibrate: Real-time Passive Non-line-of-sight Tracking_

## Environment

Create a new environment and install dependencies with `requirement.txt`:

```shell
conda create -n NLOS_Tracking

conda activate NLOS_Tracking

conda install --file requirements.txt
```

## Data Prepreation

The NLOS-Track dataset can be downloaded from [kaggle](https://www.kaggle.com/datasets/againstentropy1/nlos-track).

The file structure in project root should be as follow:

```
project_root
|   README.md
|   requirements.txt
|   train.py
+---data
+---utils
+---configs
|   ...
+---dataset
    +---render
    |   +---0000
    |   |      configs.yaml
    |   |      route.mat
    |   |      video_128.npy
    |   |      001.png
    |   |      002.png
    |   |      ...
    |   +---0001
    |       ...
    +---real-shot
        +---0000
        |      route.mat
        |      video_128.npy
        +---0001
            ...
```

### Data Loading and Visualization

Follow the code blocks in `data_playground.ipynb` to load and visualize the dataset.

**Coming soon!**

## Train

**Before training, fill the missing items in configuration files.**

Create a new configuration file in `./configs` for training:

```shell
python train.py --cfg_file=new_cfg --model_name=PAC-Net
```

or directly use `default.yaml` by default:

```shell
python train.py --model_name=PAC-Net --pretrained -b 64 -lr_b 2.5e-4 --gpu_ids=0,1 --port=8888
```

## Test

Follow the code blocks in `test.ipynb` to test a trained model.

## Ciatation

```bibtex
@article{wang2023nlosTrack,
  author   = {Wang, Yihao and Wang, Zhigang and Zhao, Bin and Wang, Dong and Chen, Mulin and Li, Xuelong},
  title    = {Propagate And Calibrate: Real-time Passive Non-line-of-sight Tracking},
  journal  = {CVPR},
  year     = {2023},
}
```
