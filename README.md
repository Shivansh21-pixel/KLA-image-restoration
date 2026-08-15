# KLA — AI Semiconductor Image Restoration

<p align="center">
  <h1 align="center">KLA · AI Semiconductor Image Restoration</h1>
  <p align="center"><b>From noisy low-resolution sensor data to clean, high-resolution semiconductor imagery using a lightweight NAFNet-inspired restoration pipeline.</b></p>
  <p align="center">
    <a href="https://github.com/Shivansh21-pixel/KLA-image-restoration"><img src="https://img.shields.io/badge/Project-Hackathon%20Submission-111827?style=for-the-badge" alt="Project"></a>
    <img src="https://img.shields.io/badge/Task-Image%20Restoration-2563eb?style=for-the-badge" alt="Task">
    <img src="https://img.shields.io/badge/Model-NAFNet%20Lite-7c3aed?style=for-the-badge" alt="Model">
    <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch">
  </p>
</p>

---

## 🏆 Executive Summary

**KLA** is an end-to-end deep-learning image restoration system designed for a semiconductor-imaging workflow where degraded **128×128 grayscale NoisyLR arrays** must be transformed into clean **256×256 reconstructions**.

Instead of treating the task as simple image resizing, the system learns a restoration mapping from paired degraded/clean examples. The final pipeline combines a **lightweight multi-scale NAFNet-inspired architecture**, residual learning, validation-driven checkpoint selection, test-time augmentation (TTA), and prediction ensembling.

### Current verified validation result

| Metric | Result |
|---|---:|
| 🥇 **Best PSNR** | **28.7921 dB** |
| 🥇 **Best SSIM** | **0.7709** |
| **Validation L1** | **0.028807** |
| Test inputs processed | **400** |
| Final output resolution | **256×256** |

> **Important:** 28.7921 dB PSNR and 0.7709 SSIM are validation results from the recorded training run. Test-set PSNR/SSIM is not claimed because test ground truth was not available locally.

---

## 🎯 The Problem

Semiconductor imaging pipelines can contain degraded, noisy, or low-resolution observations. A naive interpolation method can make an image larger, but it does **not** recover information in a learned, data-driven way.

Our objective is:

```text
             DEGRADED INPUT
             128 × 128 × 1
                   │
                   ▼
        ┌─────────────────────┐
        │   NAFNet Lite       │
        │                     │
        │ Multi-scale encoder │
        │      +              │
        │ Gated restoration   │
        │      +              │
        │ Skip connections    │
        │      +              │
        │ Decoder / upsample  │
        └─────────────────────┘
                   │
                   ▼
             RESTORED OUTPUT
             256 × 256 × 1
```

### Why this is harder than resizing

A resize algorithm only estimates missing pixels from neighboring pixels. Our network instead learns from **NoisyLR → GT** pairs and optimizes reconstruction quality using image-restoration losses and perceptual structural metrics.

---

# 🧠 Solution Architecture

## NAFNet Lite

The main model is a compact **NAFNet-inspired encoder-decoder** implemented specifically for this project.

The architecture uses:

- Multi-scale feature extraction
- Encoder/decoder hierarchy
- NAF-style SimpleGate blocks
- Depthwise convolutions
- Group normalization
- Residual learning
- Skip connections between encoder and decoder stages
- PixelShuffle-based 2× upsampling
- Lightweight channel width configuration

### Model configuration

```yaml
name: "nafnet_lite"

in_channels: 1
out_channels: 1
width: 32

enc_blocks: [2, 2, 4]
middle_blocks: 4
dec_blocks: [2, 2, 2]

scale: 2
```

### High-level flow

```text
Input 128×128
      │
      ▼
  Intro Conv
      │
      ▼
┌───────────────┐
│ Encoder Level │ ────────┐
└───────────────┘         │ Skip
      │                    │
    Down                  │
      │                    │
┌───────────────┐         │
│ Encoder Level │ ────────┤
└───────────────┘         │ Skip
      │                    │
    Down                  │
      │                    │
┌───────────────┐         │
│ Deep Encoder  │ ────────┤
└───────────────┘         │ Skip
      │                    │
      ▼                    │
┌───────────────┐         │
│    Middle     │         │
│  NAF Blocks   │         │
└───────────────┘         │
      │                    │
      ▼                    │
   Upsample ───────────────┘
      │
      ▼
   Decoder
      │
      ▼
  Upsample ×2
      │
      ▼
256×256 Output
```

> **Implementation note:** This repository uses a lightweight NAFNet-inspired design; it is **not** the original full NAFNet implementation.

---

# 🔬 Training Strategy

The project was built as an iterative experimental pipeline rather than jumping directly to a large model.

### Phase 1 — Dataset inspection

Before training, the dataset was inspected for:

- Number of `.npy` files
- Input/target shapes
- Value ranges
- Pairing consistency
- Unexpected files

This prevented silent dataset-pairing errors from contaminating the experiment.

### Phase 2 — Baseline

A small CNN baseline was trained first to establish a reference point.

Recorded baseline performance was approximately:

| Model | PSNR | SSIM |
|---|---:|---:|
| Tiny CNN Baseline | ~27.89 dB | ~0.74 |
| **NAFNet Lite** | **28.7921 dB** | **0.7709** |

The baseline gave us a measurable starting point before introducing the more capable architecture.

### Phase 3 — NAFNet Lite

The baseline was replaced with a multi-scale NAFNet-inspired architecture capable of processing features at different spatial scales.

### Phase 4 — Validation-driven checkpointing

During training, the pipeline tracks:

- Train L1
- Validation L1
- Validation PSNR
- Validation SSIM
- Best PSNR checkpoint
- Latest checkpoint

The best model is stored at:

```text
checkpoints_nafnet/best_model.pth
```

### Phase 5 — Test-time inference

The best validation checkpoint was used to process the complete test set:

```text
400 / 400 images processed
```

### Phase 6 — TTA

A second prediction stream was generated using test-time augmentation. The purpose is to reduce prediction variance and make the final reconstruction less dependent on a single input orientation/transformation.

### Phase 7 — Ensemble

Normal and TTA predictions were combined:

```text
Final = 0.5 × Normal Prediction
      + 0.5 × TTA Prediction
```

This produced a final ensemble directory containing **400 restored arrays**.

---

# 📊 Results

## Validation performance

```text
Best Validation PSNR : 28.7921 dB
Best Validation SSIM : 0.7709
Validation L1        : 0.028807
```

The best validation checkpoint was reached at the end of the recorded 50-epoch run.

### Why PSNR + SSIM?

**PSNR** measures pixel-level reconstruction fidelity. Higher is better.

**SSIM** measures structural similarity and is useful when evaluating whether edges, patterns, and local image structure are preserved.

Using both gives a more informative view than relying on only one number.

---

# 🧪 Final Prediction Pipeline

```text
                 400 Test Images
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Normal Inference          TTA Inference
             │                     │
             │                     │
             └──────────┬──────────┘
                        ▼
                    Ensemble
                        │
                        ▼
              400 Final Predictions
                        │
                        ▼
                 256 × 256 .npy
```

### Example ensemble output statistics

For `000000.npy`:

```text
Shape : (256, 256)
Min   : 0.015758
Max   : 0.949072
Mean  : 0.661427
```

These statistics are included as a sanity check for the generated output and are **not** a substitute for ground-truth scoring.

---

# 📁 Repository Structure

```text
KLA-image-restoration/
│
├── configs/
│   ├── dataset.yaml          # Dataset configuration
│   ├── model.yaml            # NAFNet Lite configuration
│   └── train.yaml            # Training configuration
│
├── datasets/
│   └── npy_dataset.py        # Paired NumPy dataset loader
│
├── models/
│   ├── nafnet_lite.py        # Main restoration network
│   └── tiny_baseline.py      # Baseline model
│
├── losses/
│   └── ...                   # Loss utilities
│
├── utils/
│   ├── checkpoint.py         # Checkpoint helpers
│   ├── metrics.py            # PSNR / SSIM utilities
│   └── seed.py               # Reproducibility helpers
│
├── scripts/
│   ├── inspect_dataset.py    # Dataset inspection
│   ├── benchmark_inference.py
│   └── metrics.py
│
├── docs/
│   └── ppt_content.md        # Presentation material
│
├── reports/
│   └── dataset_report.json   # Dataset inspection report
│
├── train.py                  # Training pipeline
├── evaluate.py               # Evaluation/inference utility
├── inference.py              # Inference entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/Shivansh21-pixel/KLA-image-restoration.git
cd KLA-image-restoration
```

## 2. Create environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure dataset paths

Edit:

```text
configs/dataset.yaml
```

with the location of your local dataset.

## 5. Inspect the dataset

```bash
python scripts/inspect_dataset.py --config configs/dataset.yaml
```

This should be done before expensive training.

## 6. Train

```bash
python train.py
```

## 7. Resume interrupted training

The training pipeline saves a latest checkpoint, allowing training to continue after interruption:

```bash
python train.py --resume
```

## 8. Run inference

```powershell
python inference.py --input_dir "PATH_TO_TEST_NOISYLR" --output_dir "results/final_nafnet" --checkpoint "checkpoints_nafnet/best_model.pth"
```

---

# 🧩 Data Format

The pipeline works with NumPy `.npy` arrays.

Expected structure:

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

Typical dimensions used in the experiment:

```text
NoisyLR : 128 × 128
GT      : 256 × 256
```

The dataset itself is intentionally **not committed to GitHub**.

---

# 🛡️ Engineering & Reproducibility

The repository separates configuration, data loading, model architecture, training, evaluation, and utility code so experiments can be reproduced without rewriting the pipeline.

The training configuration controls:

- Random seed
- Batch size
- Epoch count
- Patch size
- Learning rate
- Validation split
- Number of workers
- Mixed precision setting
- Checkpoint frequency

The pipeline also maintains separate **best** and **latest** checkpoints, making experiments safer to resume.

---

# 💡 Key Design Decisions

### 1. Start with a baseline

A baseline provides a measurable reference and prevents architectural changes from being evaluated without context.

### 2. Use residual learning

The network predicts a correction relative to the input rather than treating restoration as an entirely unrelated image-generation problem.

### 3. Use multi-scale features

Image degradation can affect both fine details and broader structures. Multi-scale processing gives the model access to different receptive-field sizes.

### 4. Validate continuously

Training loss alone is not enough. PSNR and SSIM are tracked on held-out data to identify the strongest checkpoint.

### 5. Use TTA and ensembling at inference

When multiple reasonable predictions are available, averaging them can provide a more stable final output than relying on a single inference path.

---

# 🚧 Limitations & Next Improvements

This is an actively developed hackathon system. The most important next improvements would be:

- Stronger loss design combining pixel and structural objectives
- More extensive augmentation
- Larger/deeper NAFNet configuration if compute permits
- More TTA variants
- Objective test-set scoring when GT becomes available
- Visual benchmark panels for representative samples
- Inference-time benchmarking and memory profiling
- Experiment tracking across model configurations

Being explicit about these limitations is intentional: the reported metrics are measured results, not fabricated claims.

---

# 🏁 Hackathon Story

This project follows a complete **AI engineering loop**:

```text
        Inspect Data
             ↓
       Build Baseline
             ↓
       Train & Validate
             ↓
       Analyze Metrics
             ↓
     Upgrade Architecture
             ↓
       Select Best Model
             ↓
        Run Inference
             ↓
          Apply TTA
             ↓
         Ensemble
             ↓
       Final Predictions
```

The focus is therefore not only on having a neural network, but on building a **reproducible, measurable and competition-ready restoration pipeline**.

---

# 📌 Current Status

| Component | Status |
|---|:---:|
| Dataset inspection | ✅ |
| Paired `.npy` loader | ✅ |
| CNN baseline | ✅ |
| NAFNet Lite | ✅ |
| Training pipeline | ✅ |
| Resume training | ✅ |
| PSNR / SSIM validation | ✅ |
| Best checkpoint selection | ✅ |
| 400-image inference | ✅ |
| Test-time augmentation | ✅ |
| Ensemble predictions | ✅ |
| Final visual benchmark panel | 🔄 |
| Test-GT objective scoring | ⏳ |

---

# 👤 Team

**KLA Hackathon Project**

Built with **Python + PyTorch** for AI-based semiconductor image restoration.

---

## ⭐ If you find the project interesting

Give the repository a ⭐ and explore the implementation.

**Repository:** https://github.com/Shivansh21-pixel/KLA-image-restoration
