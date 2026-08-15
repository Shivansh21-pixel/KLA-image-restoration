import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from models.nafnet_lite import build_model


def tta_predict(model, x):
    """
    Test-Time Augmentation.

    4 versions:
    1. Original
    2. Horizontal flip
    3. Vertical flip
    4. Horizontal + vertical flip

    All predictions are converted back to the original
    orientation and averaged.
    """

    predictions = []

    # --------------------------------------------------------
    # 1. Original
    # --------------------------------------------------------
    y = model(x)
    predictions.append(y)

    # --------------------------------------------------------
    # 2. Horizontal flip
    # --------------------------------------------------------
    x_h = torch.flip(x, dims=[-1])
    y_h = model(x_h)
    y_h = torch.flip(y_h, dims=[-1])
    predictions.append(y_h)

    # --------------------------------------------------------
    # 3. Vertical flip
    # --------------------------------------------------------
    x_v = torch.flip(x, dims=[-2])
    y_v = model(x_v)
    y_v = torch.flip(y_v, dims=[-2])
    predictions.append(y_v)

    # --------------------------------------------------------
    # 4. Horizontal + Vertical flip
    # --------------------------------------------------------
    x_hv = torch.flip(x, dims=[-2, -1])
    y_hv = model(x_hv)
    y_hv = torch.flip(y_hv, dims=[-2, -1])
    predictions.append(y_hv)

    # --------------------------------------------------------
    # Average predictions
    # --------------------------------------------------------
    return torch.stack(predictions, dim=0).mean(dim=0)


def main():

    # ============================================================
    # Arguments
    # ============================================================

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input_dir",
        required=True
    )

    ap.add_argument(
        "--output_dir",
        required=True
    )

    ap.add_argument(
        "--checkpoint",
        default="checkpoints_nafnet/best_model.pth"
    )

    ap.add_argument(
        "--model_config",
        default="configs/model.yaml"
    )

    ap.add_argument(
        "--tta",
        action="store_true",
        help="Use 4-way test-time augmentation"
    )

    args = ap.parse_args()

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("NAFNet INFERENCE")
    print("=" * 60)

    print("Device:", device)
    print("TTA:", args.tta)

    # ============================================================
    # Model config
    # ============================================================

    with open(args.model_config, "r") as f:
        cfg = yaml.safe_load(f)

    # ============================================================
    # Build model
    # ============================================================

    model = build_model(cfg).to(device)

    # ============================================================
    # Load checkpoint
    # ============================================================

    print("Loading checkpoint:")
    print(args.checkpoint)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device
    )

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)

    model.eval()

    print("Checkpoint loaded successfully!")

    # ============================================================
    # Input / Output
    # ============================================================

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        input_dir.glob("*.npy")
    )

    print("Found", len(files), "input files")

    # ============================================================
    # Inference
    # ============================================================

    with torch.no_grad():

        for i, p in enumerate(files, start=1):

            # ----------------------------------------------------
            # Load input
            # ----------------------------------------------------

            a = np.load(p).astype(np.float32)

            if a.ndim == 2:
                a = a[None, ...]

            x = torch.from_numpy(a).unsqueeze(0)
            x = x.to(device)

            # ----------------------------------------------------
            # Prediction
            # ----------------------------------------------------

            if args.tta:
                y = tta_predict(model, x)
            else:
                y = model(x)

            # ----------------------------------------------------
            # Remove batch
            # ----------------------------------------------------

            y = y.squeeze(0)

            # ----------------------------------------------------
            # CPU
            # ----------------------------------------------------

            y = y.cpu().numpy()

            # ----------------------------------------------------
            # Remove channel
            # ----------------------------------------------------

            if y.shape[0] == 1:
                y = y[0]

            # ----------------------------------------------------
            # Save
            # ----------------------------------------------------

            np.save(
                output_dir / p.name,
                y
            )

            if i % 100 == 0:
                print(
                    f"Processed {i}/{len(files)}"
                )

    # ============================================================
    # Done
    # ============================================================

    print()
    print("=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)

    print(
        f"Restored {len(files)} files to:"
    )

    print(output_dir)


if __name__ == "__main__":
    main()