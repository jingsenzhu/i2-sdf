**News**

- `04/04/2023` dataset preview release: 2 synthetic scenes available
- `15/04/2023` code release: 3D reconstruction and novel view synthesis part
- `21/04/2023` dataset release: real data

**TODO**

- [ ] Full dataset release
- [x] Code release for 3D reconstruction and novel view synthesis
- [ ] Code release for intrinsic decomposition and scene editing

**Dataset released**

- Synthetic: `kitchen_0`, `bedroom_relight_0`, `bedroom_0`, more scenes to be released
- Real: `inria_livingroom`, `nisr_livingroom`, `nisr_coffee_shop_0`, `nisr_coffee_shop_1`, release complete

# I<sup>2</sup>-SDF: Intrinsic Indoor Scene Reconstruction and Editing via Raytracing in Neural SDFs (CVPR 2023)

### [Project Page](https://jingsenzhu.github.io/i2-sdf/) | [Paper](https://arxiv.org/abs/2303.07634) | [Dataset](https://mega.nz/folder/jdhDnTqL#Ija678SU2Va_JJOiwqmdEg)

## Setup

### Installation

```
conda env create -f environment.yml
conda activate i2sdf
```

### Data preparation

Download our synthetic dataset and extract them into `data/synthetic`. If you want to run on your customized dataset, we provide a brief introduction to our data convention [here](DATA_CONVENTION.md).

## Dataset

We provide a high-quality synthetic indoor scene multi-view dataset, with ground truth camera pose and geometry annotations. See [HERE](DATA_CONVENTION.md) for data conventions. Click [HERE](https://mega.nz/folder/jdhDnTqL#Ija678SU2Va_JJOiwqmdEg) to download.

## 3D Reconstruction and Novel View Synthesis

### Training

```
python main_recon.py --conf config/<config_file>.yml --scan_id <scan_id> -d <gpu_id> -v <version>
```

Note: `config/synthetic.yml` doesn't contain light mask network, while `config/synthetic_light_mask.yml` contains.

If you run out of GPU memory, try to reduce the `split_n_pixels` (i.e. validation batch size), `batch_size` in the config. The default parameters are evaluated under RTX A6000 (48GB). For RTX 3090 (24GB), try to set `split_n_pixels` 5000.

### Evaluation

#### Novel view synthesis

```
python main_recon.py --conf config/<config_file>.yml --scan_id <scan_id> -d <gpu_id> -v <version> --test [--is_val] [--full]
```

The optional flag `--is_val` evaluates on the validation set instead of training set, `--full` produces full-resolution rendered images without downsampling.

#### View Interpolation

```
python main_recon.py --conf config/<config_file>.yml --scan_id <scan_id> -d <gpu_id> -v <version> --test --test_mode interpolate --inter_id <view_id_0> <view_id_1> [--full]
```

Generates a view interpolation video between 2 views. Requires `ffmpeg` being installed.

The number of frames and frame rate of the video can be specified by options.

#### Mesh Extraction

```
python main_recon.py --conf config/<config_file>.yml --scan_id <scan_id> -d <gpu_id> -v <version> --test --test_mode mesh
```

## Intrinsic Decomposition and Scene Editing

**Brewing🍺, code coming soon.**

## Citation

If you find our work is useful, please consider cite:

```
@inproceedings{zhu2023i2sdf,
    title = {I$^2$-SDF: Intrinsic Indoor Scene Reconstruction and Editing via Raytracing in Neural SDFs},
    author = {Jingsen Zhu and Yuchi Huo and Qi Ye and Fujun Luan and Jifan Li and Dianbing Xi and Lisha Wang and Rui Tang and Wei Hua and Hujun Bao and Rui Wang},
    booktitle = {CVPR},
    year = {2023}
}
```

## Acknowledgement

- This repository is built upon [Pytorch lightning](https://lightning.ai/).
- Thanks to Lior Yariv for her excellent work [VolSDF](https://lioryariv.github.io/volsdf/).
- Thanks to [Scalable-NISR](https://xchaowu.github.io/papers/scalable-nisr/) team for providing their real-world dataset.
