<div align="center">

# KLA — AI-Based Semiconductor Image Restoration

### SEMICON India Hackathon 2026 · Team WAYAN-X

**128×128 NoisyLR → 256×256 Structurally Faithful Reconstruction**

A lightweight deep-learning restoration pipeline designed for degraded semiconductor inspection imagery.

</div>

---

## 🚀 Overview

Semiconductor inspection systems operate under strict constraints on resolution, acquisition time, noise, and throughput. Low-resolution inspection scans can contain noise while simultaneously losing the fine structural information required to distinguish narrow circuit features and potential defects.

**WAYAN-X** addresses this problem as a learned image-restoration task:

```text
                 DEGRADED INPUT
                    128 × 128
                       │
                       ▼
              ┌─────────────────┐
              │  NAFNet-Lite    │
              │  Restoration    │
              │    Network      │
              └─────────────────┘
                       │
                       ▼
               256 × 256 OUTPUT
                       │
                       ▼
          Restored Semiconductor Image
```

Instead of relying on fixed interpolation such as bicubic or Lanczos, the proposed system learns the transformation from paired degraded and clean semiconductor images.

The final pipeline combines:

- A lightweight CNN baseline for establishing a reference point
- **NAFNet-inspired lightweight encoder–decoder architecture**
- Residual learning
- SimpleGate-based feature transformation
- Depthwise convolutions for efficient spatial processing
- PixelShuffle-based upsampling
- Skip connections for structural preservation
- L1 reconstruction loss
- AdamW optimization
- Cosine learning-rate scheduling
- Flip-based Test-Time Augmentation (TTA)
- Offline normal + TTA prediction ensemble

---

# 🏆 Key Results

The best NAFNet-Lite checkpoint achieved the following results on the held-out validation split:

| Metric | Result |
|---|---:|
| **Validation PSNR** | **28.7921 dB** |
| **Validation SSIM** | **0.7709** |
| Validation L1 | 0.028807 |
| Input Resolution | 128 × 128 |
| Output Resolution | 256 × 256 |
| Training Pairs | 3,200 |
| Held-out Test Images | 400 |
| Model Parameters | 2,679,457 |
| Training Epochs | 50 |

> **Important:** PSNR/SSIM values above are validation metrics. Ground truth for the 400 held-out test images was not available locally, so no test-set PSNR/SSIM is claimed.

---

# 🧠 Why NAFNet-Lite?

A conventional interpolation pipeline only estimates missing pixels using a fixed mathematical kernel.

```text
128×128
  │
  ▼
Bicubic / Lanczos
  │
  ▼
256×256
```

This does not learn the characteristics of semiconductor structures or the noise distribution present in the training data.

Our approach instead learns:

```text
NoisyLR + Learned Structural Representation
                    │
                    ▼
             Restoration Model
                    │
                    ▼
             Clean Reconstruction
```

The network is trained directly on paired **NoisyLR / Ground Truth** samples, allowing it to learn both noise suppression and structural reconstruction.

---

# 🏗️ Architecture

The main model is implemented in:

```text
models/nafnet_lite.py
```

and configured through:

```text
configs/model.yaml
```

The model is **NAFNet-inspired**, rather than a direct reproduction of the original NAFNet paper.

### Configuration

```text
width          = 32
enc_blocks     = (2, 2, 4)
middle_blocks  = 4
dec_blocks     = (2, 2, 2)
scale          = 2
```

### High-Level Architecture

```text
                         INPUT
                      128 × 128 × 1
                            │
                            ▼
                       Intro Conv
                            │
                            ▼
                  ┌─────────────────┐
                  │    Encoder 1    │
                  │  2 × NAFBlock   │
                  │     32 ch       │
                  └────────┬────────┘
                           │
                       Downsample
                           │
                           ▼
                  ┌─────────────────┐
                  │    Encoder 2    │
                  │  2 × NAFBlock   │
                  │     64 ch       │
                  └────────┬────────┘
                           │
                       Downsample
                           │
                           ▼
                  ┌─────────────────┐
                  │    Encoder 3    │
                  │  4 × NAFBlock   │
                  │    128 ch       │
                  └────────┬────────┘
                           │
                       Downsample
                           │
                           ▼
                  ┌─────────────────┐
                  │     Middle      │
                  │  4 × NAFBlock   │
                  │    256 ch       │
                  └────────┬────────┘
                           │
                       Upsample
                           │
                           ▼
                  ┌─────────────────┐
                  │    Decoder 1    │
                  │  2 × NAFBlock   │
                  │    128 ch       │
                  └────────┬────────┘
                           │
                       Upsample
                           │
                           ▼
                  ┌─────────────────┐
                  │    Decoder 2    │
                  │  2 × NAFBlock   │
                  │     64 ch       │
                  └────────┬────────┘
                           │
                       Upsample
                           │
                           ▼
                  ┌─────────────────┐
                  │    Decoder 3    │
                  │  2 × NAFBlock   │
                  │     32 ch       │
                  └────────┬────────┘
                           │
                           ▼
                      Ending Conv
                           │
                           ▼
                     256 × 256 × 1
```

The decoder performs progressive 2× PixelShuffle-based upsampling, resulting in the required **2× spatial resolution increase** from 128×128 to 256×256.

---

# 🔬 NAFBlock-Lite

Each restoration block uses two residual branches.

```text
                 Input
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
      Norm + PW1        Norm + PW1
          │                 │
          ▼                 ▼
      SimpleGate        SimpleGate
          │                 │
          ▼                 ▼
   Depthwise Conv         PW Conv
          │                 │
          ▼                 ▼
        PW2 Conv          Output
          │
          ▼
      β residual
          │
          └──────┐
                 ▼
               Merge
                 │
                 ▼
             γ residual
                 │
                 ▼
               Output
```

### Main Components

| Component | Role |
|---|---|
| **SimpleGate** | Lightweight feature gating without an activation function |
| **Depthwise Conv** | Efficient spatial feature extraction |
| **Pointwise Conv** | Channel-wise feature transformation |
| **GroupNorm** | Stable normalization with small batch sizes |
| **Residual Scaling (β, γ)** | Controls contribution of learned residual branches |
| **Skip Connections** | Preserve structural information across the encoder-decoder |
| **PixelShuffle** | Efficient spatial upsampling |

The combination keeps the model substantially lighter than a large restoration network while retaining enough capacity to learn complex image structures.

---

# ⚙️ Training Strategy

Training configuration is defined in:

```text
configs/train.yaml
```

and the training loop is implemented in:

```text
train.py
```

| Configuration | Value |
|---|---|
| Loss | `L1Loss` |
| Optimizer | AdamW |
| Learning Rate | `0.0002` |
| Weight Decay | `1e-4` |
| LR Scheduler | CosineAnnealingLR |
| Batch Size | 2 |
| Epochs | 50 |
| Patch Size | 128 |
| Train / Validation Split | 90% / 10% |
| Random Seed | 42 |
| Mixed Precision | CUDA AMP |
| Horizontal Flip | ✓ |
| Vertical Flip | ✓ |
| 90° Rotations | ✓ |

### Why L1?

L1 loss was selected as the reconstruction objective because it directly penalizes pixel-level deviation while being less sensitive to large individual errors than squared-error objectives.

```text
Prediction ─────┐
                ├── L1 Loss ──► Optimization
Ground Truth ───┘
```

---

# 📊 Validation Metrics

The model is evaluated after every epoch.

### PSNR

Peak Signal-to-Noise Ratio measures pixel-level reconstruction fidelity.

**Higher is better.**

### SSIM

Structural Similarity Index evaluates similarity in luminance, contrast, and structural information.

This is particularly relevant for semiconductor inspection because preserving edges and structural patterns is important beyond raw pixel similarity.

### Best Checkpoint

The best model is selected using:

```text
Highest Validation PSNR
```

rather than simply using the final epoch.

Final best validation result:

```text
PSNR : 28.7921 dB
SSIM : 0.7709
L1   : 0.028807
```

---

# 🆚 Baseline vs NAFNet-Lite

A lightweight Tiny CNN baseline was also implemented to establish a reference point under the same general training setup.

| Model | Parameters | Validation PSNR | Validation SSIM |
|---|---:|---:|---:|
| Tiny CNN Baseline | ~370K | ≈27.89 dB | ≈0.74 |
| **NAFNet-Lite** | **2,679,457** | **28.7921 dB** | **0.7709** |

The NAFNet-Lite model provides an approximately **0.9 dB PSNR improvement** over the baseline reference.

> Baseline values are approximate reference values from the same validation setup and are not presented as competition test-set scores.

---

# 🔄 Test-Time Augmentation

The final inference pipeline supports 4-way flip-based TTA through:

```bash
python evaluate.py --tta
```

The model evaluates the input under four transformations:

```text
Original
Horizontal Flip
Vertical Flip
Horizontal + Vertical Flip
```

Each prediction is transformed back to the original orientation and averaged.

```text
          ┌── Original ───────────┐
Input ────┼── Horizontal Flip ────┤
          ├── Vertical Flip ──────┤──► Average
          └── Both Flips ─────────┘
```

This reduces sensitivity to image orientation and can provide a more stable prediction.

---

# 🧩 Final Prediction Ensemble

For the final prediction artifacts, we additionally generated an offline ensemble between:

```text
Normal Prediction
        +
4-way TTA Prediction
```

using:

```text
Final Prediction
=
0.5 × Normal
+
0.5 × TTA
```

The resulting 400 predictions are stored in:

```text
results/ensemble/
```

### Important

The ensemble is an **offline post-processing step**. It is not currently exposed as an `--ensemble` argument in `evaluate.py`.

Because ground truth for the 400 test images is unavailable locally, we do **not** claim that the ensemble improves test PSNR/SSIM.

---

# 📦 Submission Artifacts

The repository contains the main artifacts required to reproduce and inspect the solution.

```text
checkpoints_nafnet/
└── best_model.pth
```

### Model checkpoint

```text
checkpoints_nafnet/best_model.pth
```

Best checkpoint selected using validation PSNR.

### Final predictions

```text
results/ensemble/
├── 000000.npy
├── 000001.npy
├── ...
└── 000399.npy
```

Total:

```text
400 restored predictions
```

Each prediction is stored as a NumPy array with shape:

```text
256 × 256
```

---

# 🗂️ Dataset

The pipeline operates on paired NumPy arrays.

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

### Dataset Characteristics

| Split | Input | Target | Samples |
|---|---|---|---:|
| Training | 128×128 | 256×256 | 3,200 |
| Validation | 128×128 | 256×256 | 320 |
| Test | 128×128 | — | 400 |

The test set does not include locally available ground truth.

The dataset itself is **not bundled with this repository**.

---

# 🔁 Reproducibility

## 1. Clone

```bash
git clone git@github.com:Shivansh21-pixel/KLA-image-restoration.git
cd KLA-image-restoration
```

## 2. Create Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Dataset

Update:

```text
configs/dataset.yaml
```

with the local training and test dataset paths.

---

# 🏋️ Training

Run:

```bash
python train.py
```

To resume an interrupted training run:

```bash
python train.py --resume
```

Checkpoints are written to:

```text
checkpoints_nafnet/
├── best_model.pth
└── latest_model.pth
```

---

# 🔎 Inference

Run standard inference:

```bash
python evaluate.py ^
  --input_dir PATH_TO_TEST_NOISYLR ^
  --output_dir results/normal ^
  --checkpoint checkpoints_nafnet/best_model.pth
```

For 4-way TTA:

```bash
python evaluate.py ^
  --input_dir PATH_TO_TEST_NOISYLR ^
  --output_dir results/tta ^
  --checkpoint checkpoints_nafnet/best_model.pth ^
  --tta
```

### Windows PowerShell

If using PowerShell, the commands can also be written on one line:

```powershell
python evaluate.py --input_dir "PATH_TO_TEST_NOISYLR" --output_dir "results/normal" --checkpoint "checkpoints_nafnet/best_model.pth"
```

TTA:

```powershell
python evaluate.py --input_dir "PATH_TO_TEST_NOISYLR" --output_dir "results/tta" --checkpoint "checkpoints_nafnet/best_model.pth" --tta
```

---

# 📁 Repository Structure

```text
KLA-image-restoration/
│
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── datasets/
│   ├── __init__.py
│   └── npy_dataset.py
│
├── models/
│   ├── __init__.py
│   ├── nafnet_lite.py
│   └── tiny_baseline.py
│
├── losses/
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── checkpoint.py
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
├── checkpoints_nafnet/
│   └── best_model.pth
│
├── results/
│   └── ensemble/
│       ├── 000000.npy
│       ├── ...
│       └── 000399.npy
│
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 💡 Engineering Highlights

### Reproducibility

- Fixed random seed for dataset splitting
- YAML-based configuration
- Deterministic training setup where applicable
- Checkpoint resume support

### Training

- AdamW optimizer
- Cosine learning-rate scheduling
- AMP on CUDA
- Best-checkpoint selection using validation PSNR
- Per-epoch PSNR, SSIM and L1 evaluation

### Model

- Lightweight encoder-decoder architecture
- SimpleGate feature transformation
- Depthwise convolution
- GroupNorm
- Residual learning
- Skip connections
- PixelShuffle upsampling

### Inference

- Standalone evaluation pipeline
- CPU/CUDA support
- 4-way flip TTA
- Offline normal + TTA ensemble
- NumPy `.npy` input/output

---

# ⚡ Design Philosophy

The solution was designed around three constraints:

```text
          STRUCTURAL FIDELITY
                  ▲
                  │
                  │
    COMPUTE ◄─────┼─────► ROBUSTNESS
                  │
                  ▼
             FAST ITERATION
```

Rather than maximizing model size, the objective was to find a practical balance between:

**Restoration Quality × Model Complexity × Training Efficiency**

The resulting NAFNet-Lite model contains approximately **2.68M parameters** while achieving **28.7921 dB validation PSNR**.

---

# 🔬 Limitations & Future Improvements

The current implementation leaves several directions open for further improvement.

### 1. Structural / Perceptual Loss

The current training objective is L1-only.

Potential future direction:

```text
L_total =
λ1 × L1
+
λ2 × Structural Loss
+
λ3 × Perceptual Loss
```

### 2. Larger Model Capacity

The width and block depth could be increased if additional compute and training time are available.

### 3. Advanced TTA

The current TTA uses four flip transformations.

Future experiments could investigate:

- Multi-scale TTA
- Rotation-aware TTA
- Learned prediction fusion

### 4. Test-Time Validation

Actual test-set PSNR/SSIM can only be calculated when corresponding ground truth is available.

---

# 🎯 Final Takeaway

**WAYAN-X transforms degraded 128×128 semiconductor inspection imagery into 256×256 restorations using a compact, learned image-restoration pipeline.**

The system combines:

```text
Paired Training Data
        │
        ▼
NAFNet-Inspired Architecture
        │
        ├── SimpleGate
        ├── Depthwise Convolution
        ├── Residual Learning
        ├── Skip Connections
        └── PixelShuffle
        │
        ▼
Validation-Driven Checkpoint Selection
        │
        ▼
4-Way Test-Time Augmentation
        │
        ▼
Normal + TTA Ensemble
        │
        ▼
400 Final 256×256 Predictions
```

### Best validated performance

**28.7921 dB PSNR · 0.7709 SSIM**

on the held-out validation split.

---

<div align="center">

# WAYAN-X

### KLA — AI-Based Semiconductor Image Restoration

**SEMICON India Hackathon 2026**

*Restoring the structure that matters.*

</div>
