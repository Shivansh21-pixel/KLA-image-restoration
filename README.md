# KLA Semiconductor Image Restoration

A practical PyTorch pipeline for restoring degraded grayscale semiconductor images stored as `.npy` arrays.

## Dataset expected

```text
train/
  train/
    NoisyLR/
      *.npy
    GT/
      *.npy

Test_NoisyLR/
  NoisyLR/
    *.npy
```

Do not assume that NoisyLR and GT filenames are identical. Run the inspection script first.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Edit `configs/dataset.yaml` with your local dataset paths.

## First run

```bash
python scripts/inspect_dataset.py --config configs/dataset.yaml
```

Then run a tiny sanity training before full training:

```bash
python train.py --config configs/train.yaml
```

The default config is intentionally small for a first smoke test. Increase epochs/batch size after the pipeline is verified.

## Evaluation

```bash
python evaluate.py --input_dir "PATH_TO_TEST_NOISYLR" --output_dir "results/restored"
```

## Important

No benchmark values are fabricated. PSNR/SSIM/LPIPS, runtime, and model size must be measured on the real dataset.
