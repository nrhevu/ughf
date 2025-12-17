# Uncertainty Guided Hierarchical Frequency-domain Transformer (UGHF)

UGHF is a PyTorch Lightning implementation of the **Uncertainty Guided Hierarchical Frequency-domain Transformer** paper for image restoration tasks such as non-blind deblurring. It combines frequency-domain reasoning with transformer-based modeling and an uncertainty-aware loss to recover sharp images from degraded inputs.

## Highlights
- CNNs emphasize high-frequency edges while ViTs lean toward smooth low-frequency content, leaving a spectrum gap and causing artifacts.
- UGHF (HFDT) bridges the gap with a dual-domain interaction block that merges FFT-derived global cues with lightweight spatial convolutions.
- An uncertainty-guided prior spotlights severely degraded pixels, enabling consistent restoration across rain, blur, and other challenging corruptions.

## Requirements
Create a new Python environment (3.8+) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Preparation
Organize datasets using parallel `blur` and `sharp` directories for both training and validation:

```
datasets/
├── train
│   ├── blur
│   └── sharp
└── val
    ├── blur
    └── sharp
```

Each folder must contain aligned image pairs with matching filenames (e.g., `blur/img001.png` ↔ `sharp/img001.png`). Update `train_pl.py` if you want to use a different layout.

## Training
Launch training with:

```bash
python train_pl.py \
  --train_dir /path/to/datasets/train \
  --val_dir /path/to/datasets/val \
  --batch_size 8 \
  --devices auto
```

Important flags:
- `--pretrain_weights`: directory to store checkpoints.
- `--precision`: set to 16 for mixed precision on supported GPUs.
- `--num_workers`: dataloader workers; tune based on CPU cores.

<!-- ## Evaluation / Inference
After training, run validation or test inference:

```bash
python eval_pl.py \
  --ckpt path/to/checkpoint.ckpt \
  --input_dir /path/to/blur/images \
  --output_dir ./restored
```

`eval_pl.py` saves restored images and logs PSNR/SSIM metrics when ground truth is available.

## Tips
- Follow PyTorch Lightning logging callbacks to monitor losses and learning rate.
- Use `--resume_from_checkpoint` to continue interrupted training.
- For new datasets, adjust normalization and data augmentations in `datasets/*.py`. -->

## Citation
Please cite the original UGHF paper if you use this code in your research.
```cite
@article{article,
    author = {Shao, Mingwen and Qiao, Yuanjian and Meng, Deyu and Zuo, Wangmeng},
    year = {2023},
    month = {03},
    pages = {110306},
    title = {Uncertainty-guided hierarchical frequency domain Transformer for image restoration},
    volume = {263},
    journal = {Knowledge-Based Systems},
    doi = {10.1016/j.knosys.2023.110306}
}
```
