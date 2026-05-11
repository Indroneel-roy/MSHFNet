<div align="center">

# MSHFNet: Multi-Scale Hybrid Fusion Network
### Precise Multi-Organ Abdominal CT Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/Institution-UCER-orange)](https://www.ucer.ac.in/)

**Indroneel Roy** · United College of Engineering and Research

</div>

---

## Overview

**MSHFNet** (Multi-Scale Hybrid Fusion Network) is a dual-pathway medical image segmentation architecture that bridges the representational strengths of CNNs and Vision Transformers through bidirectional cross-attention. Rather than simple feature concatenation or addition, our novel **Cross-Attention Fusion Module (CAFM)** enables each encoder branch to selectively query the other at four hierarchical scales — combining local spatial precision with global contextual understanding in a single unified framework.

<div align="center">

| Benchmark | Mean DSC ↑ | Mean HD95 ↓ | Pancreas DSC ↑ |
|:---:|:---:|:---:|:---:|
| **Synapse (8 organs)** | **79.73%** | **27.84 mm** | **81.47%** *(SOTA)* |
| **ACDC (cardiac MRI)** | **93.76%** | **1.03 mm** | — |

</div>

---

## Architecture

MSHFNet processes a single-channel CT slice of size `1×224×224` through three stages:

```
Input CT Slice (1×224×224)
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
 ┌─────────────┐                 ┌─────────────────┐
 │  ResNet-50  │                 │   Swin-Base     │
 │ CNN Encoder │                 │ Transformer Enc.│
 │  (ImageNet) │                 │  (ImageNet-22K) │
 └──────┬──────┘                 └────────┬────────┘
        │  {S1, S2, S3, S4}               │  {S1, S2, S3, S4}
        │  Ch: 256,512,1024,2048          │  Ch: 128,256,512,1024
        └──────────┬──────────────────────┘
                   ▼  (at each scale)
          ┌────────────────┐
          │  CAFM          │  ← Bidirectional Cross-Attention
          │  (256-d joint  │     CNN queries Transformer
          │   embedding)   │     Transformer queries CNN
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │ Shared Decoder │  ← UNet-style with skip connections
          │ + Deep Superv. │     3 auxiliary heads (DS1, DS2, DS3)
          └────────┬───────┘
                   ▼
         Segmentation Mask (9 classes)
```

### Key Components

**1. Dual-Pathway Hierarchical Encoder**
- **CNN Branch (ResNet-50):** Extracts local texture features at 4 scales with channel dims `{256, 512, 1024, 2048}` and spatial resolutions `{56, 28, 14, 7}`
- **Transformer Branch (Swin-Base):** Captures global long-range context via shifted-window self-attention at matching scales with channel dims `{128, 256, 512, 1024}`

**2. Cross-Attention Fusion Module (CAFM)**
Both branches are projected to a shared 256-d space, then cross-attend bidirectionally:
- **Direction 1 — CNN queries Transformer:** CNN branch gathers global context
- **Direction 2 — Transformer queries CNN:** Transformer branch gathers local spatial precision
- Outputs are concatenated, refined via `3×3` Conv + residual shortcut, and regularized with Dropout (p=0.1)

**3. Shared Decoder with Deep Supervision**
- Progressive upsampling: `S4 → S3 → S2 → S1 → 224×224`
- Three auxiliary heads at `D1`, `D2`, `D3` with loss weights `w=[1.0, 0.4, 0.2, 0.1]`
- Only the main head is used at inference

---

## Results

### Synapse Multi-Organ CT Benchmark

| Method | DSC% ↑ | HD95 ↓ | Aorta | Gallbladder | Kidney(L) | Kidney(R) | Liver | Pancreas | Spleen | Stomach |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| V-Net | 68.81 | — | 75.34 | 51.87 | 77.10 | 80.75 | 87.84 | 40.05 | 80.56 | 56.98 |
| U-Net | 74.68 | 36.87 | 84.18 | 62.84 | 79.19 | 71.29 | 93.35 | 48.23 | 84.41 | 73.92 |
| Attention U-Net | 75.57 | 36.97 | 55.92 | 63.91 | 79.20 | 72.71 | 93.56 | 49.37 | 87.19 | 74.95 |
| TransUNet | 77.48 | 31.69 | 87.23 | 63.13 | 81.87 | 77.02 | 94.08 | 55.86 | 85.08 | 75.62 |
| Swin-UNet | 79.13 | **21.55** | 85.47 | 66.53 | 83.28 | 79.61 | 94.29 | 56.58 | **90.66** | 76.60 |
| **MSHFNet (Ours)** | **79.73** | 27.84 | **88.49** | 62.16 | 77.07 | 64.35 | **94.98** | **81.47** | 82.50 | **86.30** |

### ACDC Cardiac Segmentation Benchmark

| Structure | DSC% ↑ | HD95 (mm) ↓ |
|:---|:---:|:---:|
| Right Ventricle | 95.77 | 1.00 |
| Myocardium | 90.17 | 1.00 |
| Left Ventricle | 95.35 | 1.10 |
| **Mean** | **93.76** | **1.03** |

### Ablation Study

| Configuration | CNN | Swin | CAFM | Deep Sup. | DSC% ↑ | HD95 ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Exp1: CNN Only | ✓ | ✗ | ✗ | ✓ | 71.08 | 41.54 |
| Exp2: Swin Only | ✗ | ✓ | ✗ | ✓ | 78.93 | 30.75 |
| Exp3: Dual Enc. + Simple Fusion | ✓ | ✓ | ✗ | ✗ | 77.66 | 28.41 |
| Exp4: Dual Enc. + CAFM | ✓ | ✓ | ✓ | ✗ | 78.41 | 30.04 |
| **MSHFNet (Full)** | ✓ | ✓ | ✓ | ✓ | **79.73** | **27.84** |

---

## Requirements

```bash
Python        >= 3.8
PyTorch       >= 2.0.0
CUDA          >= 11.8
torchvision   >= 0.15.0
timm          >= 0.9.0
numpy         >= 1.21.0
scipy         >= 1.7.0
SimpleITK     >= 2.1.0
einops        >= 0.6.0
tqdm          >= 4.64.0
medpy         >= 0.4.0
```

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/IndroneelRoy/MSHFNet.git
cd MSHFNet
```

### 2. Prepare the Datasets

**Synapse Multi-Organ CT**

Download the pre-processed data from the [TransUNet repository](https://github.com/Beckschen/TransUNet).
Place it in the following structure:

```
data/
└── Synapse/
    ├── train_npz/
    │   ├── case0001_slice000.npz
    │   ├── case0001_slice001.npz
    │   └── ...
    └── test_vol_h5/
        ├── case0001.npy.h5
        ├── case0002.npy.h5
        └── ...
```

**ACDC Cardiac MRI**

Download from the [official ACDC challenge website](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
Place it under `data/ACDC/`.

### 3. Download Pretrained Backbone Weights

The backbone weights are loaded automatically via `torchvision` (ResNet-50) and `timm` (Swin-Base). Alternatively, pre-download them:

```bash
# Swin-Base ImageNet-22K weights
python scripts/download_weights.py --backbone swin_base_patch4_window7_224_22k
```

### 4. Train

**Synapse dataset:**

```bash
python train.py \
    --dataset Synapse \
    --root_path data/Synapse/train_npz \
    --num_classes 9 \
    --img_size 224 \
    --max_epochs 70 \
    --batch_size 16 \
    --base_lr 1e-4 \
    --encoder_lr 1e-5 \
    --weight_decay 1e-4 \
    --deep_supervision \
    --output_dir outputs/synapse
```

**ACDC dataset:**

```bash
python train.py \
    --dataset ACDC \
    --root_path data/ACDC \
    --num_classes 4 \
    --img_size 224 \
    --max_epochs 70 \
    --batch_size 16 \
    --base_lr 1e-4 \
    --encoder_lr 1e-5 \
    --output_dir outputs/acdc
```

### 5. Test & Evaluate

```bash
python test.py \
    --dataset Synapse \
    --root_path data/Synapse/test_vol_h5 \
    --num_classes 9 \
    --img_size 224 \
    --checkpoint outputs/synapse/best_model.pth \
    --tta \
    --output_dir results/synapse
```

Output metrics (DSC and HD95 per organ) will be saved to `results/synapse/metrics.csv`.

---

## Project Structure

```
MSHFNet/
│
├── data/                        # Dataset directories (not tracked by git)
│   ├── Synapse/
│   └── ACDC/
│
├── models/
│   ├── mshfnet.py               # Full MSHFNet model
│   ├── cafm.py                  # Cross-Attention Fusion Module
│   ├── encoder_cnn.py           # ResNet-50 CNN encoder
│   ├── encoder_swin.py          # Swin-Base Transformer encoder
│   └── decoder.py               # Shared decoder with deep supervision
│
├── utils/
│   ├── dataset_synapse.py       # Synapse dataloader
│   ├── dataset_acdc.py          # ACDC dataloader
│   ├── losses.py                # Combined CE + Dice loss
│   ├── metrics.py               # DSC and HD95 computation
│   └── augmentation.py          # Online augmentation pipeline
│
├── scripts/
│   └── download_weights.py      # Backbone weight downloader
│
├── Figure/
│   ├── image1.png               # MSHFNet architecture diagram
│   ├── image2.png               # CAFM schematic
│   ├── image3.png               # Decoder diagram
│   └── comparison_figure.png    # Qualitative results
│
├── train.py                     # Training entry point
├── test.py                      # Evaluation entry point
├── requirements.txt             # Python dependencies
├── LICENSE
└── README.md
```

---

## Training Details

| Hyperparameter | Value |
|:---|:---:|
| Optimizer | AdamW |
| Base learning rate (new modules) | 1e-4 |
| Encoder learning rate (pretrained) | 1e-5 |
| LR Scheduler | Cosine Annealing |
| Min LR | 1e-6 |
| Weight decay | 1e-4 |
| Gradient clip norm | 1.0 |
| Batch size | 16 |
| Epochs | 70 |
| Input size | 224×224 |
| Loss | 0.5 × CE + 0.5 × Dice |
| Deep supervision weights | [1.0, 0.4, 0.2, 0.1] |

**Preprocessing:**
- HU clipping to `[-125, 275]`
- Per-slice mean-std normalization
- Bilinear resize to `224×224`

**Augmentation (online, 50% probability each):**
- Horizontal / vertical flip
- 90° / 180° / 270° rotation
- Brightness scaling ∈ [0.7, 1.3]
- Additive Gaussian noise (σ=0.1)
- Contrast adjustment ∈ [0.8, 1.2]
- Random crop-and-zoom ∈ [0.8, 1.0]

**Test-Time Augmentation (TTA):**
Averages softmax predictions from original, horizontal flip, and vertical flip before argmax.

---

## Computing Environment

| Component | Specification |
|:---|:---|
| GPU | NVIDIA Tesla T4 (15.6 GB VRAM) |
| Platform | Kaggle Cloud |
| Framework | PyTorch 2.10.0 |
| CUDA | 12.8 |
| CNN Backbone | ResNet-50 (torchvision, ImageNet) |
| Transformer Backbone | Swin-Base (timm, ImageNet-22K) |

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{roy2025mshfnet,
  title     = {MSHFNet: Multi-Scale Hybrid Fusion Network for
               Precise Multi-Organ Abdominal CT Segmentation},
  author    = {Roy, Indroneel},
  journal   = {arXiv preprint},
  year      = {2025},
  note      = {United College of Engineering and Research}
}
```

---

## Acknowledgements

- The Synapse dataset split and preprocessing protocol follows [TransUNet](https://github.com/Beckschen/TransUNet).
- Swin-Base backbone weights from [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
- ResNet-50 backbone from [torchvision](https://pytorch.org/vision/stable/models.html).
- The author thanks **Mr. Dharmendra Sir**, Head of Department, United College of Engineering and Research, for his constant encouragement and guidance.

---

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built with ❤️ for advancing medical image segmentation research</sub>
</div>
