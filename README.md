# KLA — AI Semiconductor Image Restoration

> **Deep-learning pipeline for restoring noisy low-resolution grayscale semiconductor images into clean 256×256 reconstructions.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-NAFNet%20Lite-purple)](#model)
[![Task](https://img.shields.io/badge/Task-Image%20Restoration-green)](#problem)

## 🧠 Problem

The input data contains **128×128 noisy low-resolution grayscale images**, while the corresponding clean ground-truth images are **256×256**.

The goal is to learn a mapping:

```text
Noisy LR image (128×128)
          │
          ▼
     NAFNet Lite
          │
          ▼
Restored image (256×256)
```

The challenge is not simply resizing the image. The model must learn to **remove degradation while reconstructing useful spatial detail**.

## 🚀 Approach

We built the pipeline incrementally:

1. **Dataset inspection** — verified array shapes, value ranges and paired samples.
2. **CNN baseline** — established a simple restoration baseline.
3. **NAFNet-inspired model** — replaced the baseline with a lightweight multi-scale encoder-decoder.
4. **Residual learning** — the network predicts a restoration correction and adds it to the input.
5. **Validation metrics** — tracked L1 loss, PSNR and SSIM during training.
6. **Test-Time Augmentation (TTA)** — generated an additional prediction using transformed inputs.
7. **Ensembling** — combined the normal and TTA predictions for the final candidate outputs.

## 🏗️ Model

### NAFNet Lite

The model is inspired by the design principles of NAFNet and is implemented specifically for this project.

Key components:

- Grayscale input/output
- Multi-scale encoder-decoder
- NAF-style gated blocks
- Depthwise convolution
- Group normalization
- Skip connections
- PixelShuffle-based upsampling
- Residual output connection
- 2× spatial reconstruction

Configuration used for the main experiment:

```yaml
width: 32
enc_blocks: [2, 2, 4]
middle_blocks: 4
dec_blocks: [2, 2, 2]
scale: 2
```

> **Note:** This repository contains a lightweight **NAFNet-inspired implementation**, not the original full NAFNet implementation.

## 📊 Validation Results

The best recorded validation checkpoint achieved:

| Metric | Best validation result |
|---|---:|
| **PSNR** | **28.7921 dB** |
| **SSIM** | **0.7709** |
| **Validation L1** | **0.028807** |

The best checkpoint is saved during training as:

```text
checkpoints_nafnet/best_model.pth
```

### Baseline vs NAFNet Lite

| Model | Validation PSNR | Validation SSIM |
|---|---:|---:|
| Tiny CNN Baseline | ~27.89 dB | ~0.74 |
| **NAFNet Lite** | **28.7921 dB** | **0.7709** |

The baseline comparison is included to show the improvement obtained by moving to the multi-scale NAFNet-inspired architecture.

## 🔬 TTA & Ensemble

For the final test pipeline, we generated:

- **Normal NAFNet prediction:** 400 images
- **TTA prediction:** 400 images
- **Ensemble prediction:** 400 images

The final ensemble is the simple average of the normal and TTA predictions:

```text
Final = 0.5 × Normal + 0.5 × TTA
```

Example final output:

```text
Shape : (256, 256)
Min   : 0.015758
Max   : 0.949072
Mean  : 0.661427
```

> Test-set PSNR/SSIM is not reported because the test ground-truth images were not available for scoring locally. The **28.7921 dB / 0.7709 SSIM** figures above are validation results.

## 📁 Project Structure

```text
KLA-image-restoration/
│
├── configs/
│   ├── dataset.yaml       # Dataset paths and pairing configuration
│   ├── model.yaml         # NAFNet Lite architecture configuration
│   └── train.yaml         # Training configuration
│
├── datasets/
│   └── npy_dataset.py     # Paired .npy dataset loader
│
├── models/
│   ├── nafnet_lite.py     # Main restoration model
│   └── tiny_baseline.py   # Baseline CNN
│
├── losses/
├── utils/
├── scripts/
│   ├── inspect_dataset.py
│   ├── benchmark_inference.py
│   └── metrics.py
│
├── docs/
├── reports/
├── train.py               # Training pipeline
├── inference.py           # Test inference
├── evaluate.py            # Model evaluation/inference utility
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Dataset Format

The pipeline expects paired NumPy arrays similar to:

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

Test_NoisyLR/
└── NoisyLR/
    ├── 000000.npy
    ├── 000001.npy
    └── ...
```

Typical shapes used in this project:

```text
NoisyLR : (128, 128)
GT      : (256, 256)
```

The actual dataset is **not included in this repository**.

## 🛠️ Installation

### 1. Clone

```bash
git clone https://github.com/Shivansh21-pixel/KLA-image-restoration.git
cd KLA-image-restoration
```

### 2. Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure dataset paths

Edit:

```text
configs/dataset.yaml
```

and point the paths to your local dataset.

## 🏋️ Training

Run the training pipeline with the project configuration:

```bash
python train.py
```

The training pipeline:

- loads paired NumPy images
- creates train/validation splits
- trains NAFNet Lite
- calculates L1 loss
- calculates PSNR and SSIM
- saves the latest checkpoint
- saves the best PSNR checkpoint

To resume from the latest saved checkpoint:

```bash
python train.py --resume
```

## 🔮 Inference

Run inference using the best NAFNet Lite checkpoint:

```powershell
python inference.py --input_dir "PATH_TO_TEST_NOISYLR" --output_dir "results/final_nafnet" --checkpoint "checkpoints_nafnet/best_model.pth"
```

The restored images are saved as `.npy` arrays with the corresponding filenames.

## 📈 Evaluation Metrics

### PSNR

Peak Signal-to-Noise Ratio measures reconstruction fidelity. Higher is better.

### SSIM

Structural Similarity measures how closely the restored image preserves structural information from the ground truth. Higher is better.

Both metrics are calculated after clamping predictions and targets to the expected `[0, 1]` image range.

## 💡 Why This Pipeline?

A simple resize operation can increase an image from 128×128 to 256×256, but it cannot learn the underlying restoration problem.

This project instead learns from paired examples:

```text
Noisy / degraded image  →  Clean ground truth
```

The model therefore learns both **restoration** and **2× reconstruction** from the training data.

## 🏆 Hackathon Takeaway

The project demonstrates a complete deep-learning restoration workflow rather than only a model definition:

**Inspect → Baseline → Train → Validate → Improve → TTA → Ensemble → Generate final predictions**

The final pipeline produces **400 restored 256×256 `.npy` predictions** for the provided test set.

## 👥 Team

Built as a hackathon project focused on practical AI-based image restoration for semiconductor imagery.

---

⭐ If this project is useful or interesting, consider starring the repository.
