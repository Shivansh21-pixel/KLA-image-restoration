# KLA — AI Semiconductor Image Restoration

<p align="center">
  <h1 align="center">KLA · AI Semiconductor Image Restoration</h1>
  <p align="center"><b>Restoring 128×128 noisy low-resolution semiconductor imagery into 256×256 high-resolution outputs.</b></p>
  <p align="center">
    <b>WAYAN-X</b>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Image%20Restoration-2563eb?style=for-the-badge" alt="Task">
  <img src="https://img.shields.io/badge/Model-NAFNet%20Lite-7c3aed?style=for-the-badge" alt="Model">
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Test%20Images-400-059669?style=for-the-badge" alt="Test Images">
</p>

---

## 🎯 The Challenge

Semiconductor imaging can produce noisy and low-resolution observations where important structures are difficult to recover. Simple interpolation can increase resolution, but it does not learn how degraded observations map to clean images.

**Our goal:** learn that mapping from paired examples and reconstruct a sharper **256×256** image from a **128×128** noisy input while preserving structural information.

```text
128×128 NoisyLR
       │
       ▼
┌──────────────────────┐
│     NAFNet Lite      │
│  Multi-scale encoder │
│  SimpleGate blocks   │
│  Skip connections    │
│  Residual learning   │
│  Decoder + upsample  │
└──────────────────────┘
       │
       ▼
256×256 Restored Image
       │
       ├── TTA inference
       │
       └── Prediction ensemble
```

---

## 🏆 Results

The strongest recorded validation checkpoint from the 50-epoch training run achieved:

| Metric | Best Validation Result |
|---|---:|
| **PSNR** | **28.7921 dB** |
| **SSIM** | **0.7709** |
| Validation L1 | **0.028807** |
| Test inputs processed | **400** |
| Output resolution | **256×256** |

### Baseline → NAFNet Lite

| Model | PSNR | SSIM |
|---|---:|---:|
| Tiny CNN Baseline | ~27.89 dB | ~0.74 |
| **NAFNet Lite** | **28.7921 dB** | **0.7709** |

> These PSNR/SSIM values are **validation results** from the recorded experiment. No test-set PSNR/SSIM is claimed because test ground truth was not available locally.

---

## 🧠 Our Approach

### 1. Establish a baseline

We first trained a compact CNN to obtain a reference point before moving to a more capable restoration architecture.

### 2. Multi-scale restoration with NAFNet Lite

The main model is a lightweight **NAFNet-inspired encoder-decoder** built for this task. It uses:

- SimpleGate-based blocks
- Depthwise convolutions
- Group normalization
- Multi-scale encoder/decoder processing
- Encoder-decoder skip connections
- Residual learning
- PixelShuffle-based upsampling

```text
Input
  │
  ▼
Intro Conv
  │
  ├─────────────── Skip ───────────────┐
  ▼                                    │
Encoder 1 → Down                      │
  │                                    │
  ├─────────────── Skip ───────────┐  │
  ▼                                │  │
Encoder 2 → Down                  │  │
  │                                │  │
  ▼                                │  │
Encoder 3 → Down                  │  │
  │                                │  │
  ▼                                │  │
Middle NAF Blocks                  │  │
  │                                │  │
  ▼                                │  │
Upsample → Decoder ←───────────────┘  │
  │                                   │
  ▼                                   │
Upsample → Decoder ←──────────────────┘
  │
  ▼
256×256 Output
```

The implementation is intentionally lightweight rather than being the original full NAFNet implementation.

### 3. Validation-driven training

The training pipeline tracks **L1, PSNR and SSIM** on a held-out validation split and saves both the latest checkpoint and the best-performing checkpoint.

```text
checkpoints_nafnet/
├── best_model.pth
└── latest_model.pth
```

### 4. Robust inference

The best checkpoint was used to restore all **400 test images**. We then explored test-time augmentation and combined normal/TTA predictions through a simple ensemble.

```text
                 Test Set
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Normal Inference      TTA Inference
          │                   │
          └─────────┬─────────┘
                    ▼
                 0.5 / 0.5
                 Ensemble
                    │
                    ▼
             400 Final .npy
```

---

## 🔬 Why This Pipeline?

**Baseline first.** We wanted a measurable reference before increasing model complexity.

**Multi-scale features.** Restoration needs both local detail and broader contextual information.

**Residual learning.** The network learns a restoration correction while retaining the useful information already present in the input.

**PSNR + SSIM.** PSNR captures pixel-level fidelity while SSIM provides a structural-quality view.

**TTA + ensemble.** Multiple inference paths can be combined to make predictions less dependent on a single transformation.

---

## 📁 Project Structure

```text
KLA-image-restoration/
│
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── datasets/
│   └── npy_dataset.py
│
├── models/
│   ├── nafnet_lite.py
│   └── tiny_baseline.py
│
├── losses/
├── utils/
│   ├── checkpoint.py
│   ├── metrics.py
│   └── seed.py
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── benchmark_inference.py
│   └── metrics.py
│
├── reports/
│   └── dataset_report.json
│
├── docs/
│   └── ppt_content.md
│
├── train.py
├── inference.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## ⚡ Run the Project

### Clone

```bash
git clone https://github.com/Shivansh21-pixel/KLA-image-restoration.git
cd KLA-image-restoration
```

### Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure

Set the local dataset paths in:

```text
configs/dataset.yaml
```

### Train

```bash
python train.py
```

### Resume training

```bash
python train.py --resume
```

### Inference

```powershell
python inference.py --input_dir "PATH_TO_TEST_NOISYLR" --output_dir "results/final_nafnet" --checkpoint "checkpoints_nafnet/best_model.pth"
```

---

## 📦 Data Format

The experiment uses paired NumPy arrays:

```text
train/
├── NoisyLR/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
│
└── GT/
    ├── 000000.npy
    ├── 000001.npy
    └── ...
```

Typical experiment dimensions:

```text
NoisyLR : 128×128
GT      : 256×256
```

The dataset itself is not included in this repository.

---

## 🚀 What We Would Improve Next

With additional compute/time, the next experiments would focus on:

- Stronger pixel + structural loss combinations
- More diverse augmentation
- Larger NAFNet configurations
- More TTA variants
- Objective scoring once test ground truth is available
- More systematic experiment tracking
- Visual benchmark panels for representative samples
- Inference speed and memory optimization

---

## 👥 Team

### WAYAN-X

**KLA Hackathon — AI Semiconductor Image Restoration**

Built with **Python, PyTorch and deep-learning based image restoration**.

---

<p align="center">
  <b>WAYAN-X</b><br>
  <sub>Restoring clarity from degraded semiconductor imagery.</sub>
</p>
