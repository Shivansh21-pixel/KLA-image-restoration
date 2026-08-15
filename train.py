import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.npy_dataset import pair_files, NpyPairedDataset
from models.nafnet_lite import build_model
from scripts.metrics import calculate_metrics


# ============================================================
# Seed
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. Arguments
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/train.yaml"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest_model.pth"
    )

    args = parser.parse_args()


    # ========================================================
    # 2. Config
    # ========================================================

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    with open(cfg["dataset_config"], "r") as f:
        dcfg = yaml.safe_load(f)

    with open(cfg["model_config"], "r") as f:
        model_cfg = yaml.safe_load(f)

    seed_all(cfg.get("seed", 42))


    # ========================================================
    # 3. Device
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DEVICE")
    print("=" * 60)
    print(device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("=" * 60)


    # ========================================================
    # 4. Dataset
    # ========================================================

    train_root = Path(dcfg["train_root"])

    noisy_dir = train_root / dcfg["noisy_dir_name"]
    gt_dir = train_root / dcfg["gt_dir_name"]

    print("Noisy directory:", noisy_dir)
    print("GT directory   :", gt_dir)

    pairs = pair_files(
        noisy_dir,
        gt_dir,
        dcfg.get("pairing", "same_stem")
    )

    print("Total paired files:", len(pairs))


    # ========================================================
    # 5. Dataset
    # ========================================================

    full_ds = NpyPairedDataset(
        pairs,
        patch_size=cfg.get("patch_size"),
        train=True
    )


    # ========================================================
    # 6. Train / Validation split
    # ========================================================

    val_fraction = cfg.get(
        "val_fraction",
        0.1
    )

    nval = max(
        1,
        int(len(full_ds) * val_fraction)
    )

    ntrain = len(full_ds) - nval

    generator = torch.Generator().manual_seed(
        cfg.get("seed", 42)
    )

    indices = torch.randperm(
        len(full_ds),
        generator=generator
    ).tolist()

    train_indices = indices[:ntrain]
    val_indices = indices[ntrain:]


    train_ds = Subset(
        full_ds,
        train_indices
    )


    # Validation must not use augmentation

    val_full_ds = NpyPairedDataset(
        pairs,
        patch_size=None,
        train=False
    )

    val_ds = Subset(
        val_full_ds,
        val_indices
    )


    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print("Total      :", len(full_ds))
    print("Train      :", len(train_ds))
    print("Validation :", len(val_ds))
    print("=" * 60)


    # ========================================================
    # 7. DataLoaders
    # ========================================================

    batch_size = cfg.get(
        "batch_size",
        2
    )

    num_workers = cfg.get(
        "num_workers",
        0
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )


    # ========================================================
    # 8. Model
    # ========================================================

    print("=" * 60)
    print("MODEL")
    print("=" * 60)

    model = build_model(
        model_cfg
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters    :", f"{total_params:,}")
    print("Trainable parameters:", f"{trainable_params:,}")

    print("=" * 60)


    # ========================================================
    # 9. Loss
    # ========================================================

    criterion = nn.L1Loss()


    # ========================================================
    # 10. Optimizer
    # ========================================================

    learning_rate = cfg.get(
        "learning_rate",
        0.0002
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )


    # ========================================================
    # 11. Scheduler
    # ========================================================

    epochs = cfg.get(
        "epochs",
        50
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )


    # ========================================================
    # 12. AMP
    # ========================================================

    use_amp = (
        cfg.get("amp", True)
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )


    # ========================================================
    # 13. Checkpoint directory
    # ========================================================

    output_dir = Path(
        cfg.get(
            "output_dir",
            "checkpoints_nafnet"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_checkpoint = (
        output_dir / "best_model.pth"
    )

    latest_checkpoint = (
        output_dir / "latest_model.pth"
    )


    # ========================================================
    # 14. Resume
    # ========================================================

    start_epoch = 1

    best_psnr = -float("inf")

    if args.resume:

        if latest_checkpoint.exists():

            print("=" * 60)
            print("RESUMING TRAINING")
            print("=" * 60)

            checkpoint = torch.load(
                latest_checkpoint,
                map_location=device
            )

            # Model
            model.load_state_dict(
                checkpoint["model"]
            )

            # Optimizer
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )

            # Scheduler
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(
                    checkpoint["scheduler"]
                )

            # AMP scaler
            if "scaler" in checkpoint:
                scaler.load_state_dict(
                    checkpoint["scaler"]
                )

            # Last completed epoch
            last_epoch = checkpoint.get(
                "epoch",
                0
            )

            start_epoch = last_epoch + 1

            # Best PSNR
            best_psnr = checkpoint.get(
                "best_psnr",
                -float("inf")
            )

            print(
                "Last completed epoch:",
                last_epoch
            )

            print(
                "Starting epoch:",
                start_epoch
            )

            print(
                "Best PSNR:",
                best_psnr
            )

            print("=" * 60)

        else:

            print(
                "WARNING: Resume requested but "
                "latest checkpoint was not found."
            )

            print(
                "Starting training from epoch 1."
            )


    # ========================================================
    # 15. Training loop
    # ========================================================

    for epoch in range(
        start_epoch,
        epochs + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}"
        )

        for x, y, _, _ in progress:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # --------------------------------------------
            # Forward
            # --------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                pred = model(x)

                if pred.shape != y.shape:

                    raise RuntimeError(
                        f"Shape mismatch: "
                        f"prediction={pred.shape}, "
                        f"target={y.shape}"
                    )

                loss = criterion(
                    pred,
                    y
                )

            # --------------------------------------------
            # Backward
            # --------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
                * x.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_total = 0.0

        psnr_values = []
        ssim_values = []


        with torch.no_grad():

            val_progress = tqdm(
                val_loader,
                desc=f"Validation {epoch}"
            )

            for x, y, _, _ in val_progress:

                x = x.to(
                    device,
                    non_blocking=True
                )

                y = y.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp
                ):

                    pred = model(x)

                    if pred.shape != y.shape:

                        raise RuntimeError(
                            f"Validation shape mismatch: "
                            f"prediction={pred.shape}, "
                            f"target={y.shape}"
                        )

                    loss = criterion(
                        pred,
                        y
                    )

                val_loss_total += (
                    loss.item()
                    * x.size(0)
                )

                metrics = calculate_metrics(
                    pred,
                    y
                )

                psnr_values.append(
                    float(metrics["psnr"])
                )

                ssim_values.append(
                    float(metrics["ssim"])
                )


        val_loss = (
            val_loss_total
            / len(val_loader.dataset)
        )

        avg_psnr = float(
            np.mean(psnr_values)
        )

        avg_ssim = float(
            np.mean(ssim_values)
        )


        # ====================================================
        # Scheduler
        # ====================================================

        scheduler.step()


        # ====================================================
        # Save latest checkpoint
        # ====================================================

        checkpoint = {

            "epoch": epoch,

            "model": model.state_dict(),

            "optimizer": optimizer.state_dict(),

            "scheduler": scheduler.state_dict(),

            "scaler": scaler.state_dict(),

            "best_psnr": best_psnr,

            "train_loss": train_loss,

            "val_loss": val_loss,

            "psnr": avg_psnr,

            "ssim": avg_ssim,

        }

        torch.save(
            checkpoint,
            latest_checkpoint
        )


        # ====================================================
        # Save best model
        # ====================================================

        if avg_psnr > best_psnr:

            best_psnr = avg_psnr

            checkpoint["best_psnr"] = best_psnr

            torch.save(
                checkpoint,
                best_checkpoint
            )

            is_best = True

        else:

            is_best = False


        # ====================================================
        # Print results
        # ====================================================

        print()
        print("=" * 60)
        print(
            f"Epoch       : {epoch}"
        )
        print(
            f"Train L1    : {train_loss:.6f}"
        )
        print(
            f"Val L1      : {val_loss:.6f}"
        )
        print(
            f"Val PSNR    : {avg_psnr:.4f}"
        )
        print(
            f"Val SSIM    : {avg_ssim:.4f}"
        )
        print(
            f"Best PSNR   : {best_psnr:.4f}"
        )

        if is_best:
            print(
                "🔥 NEW BEST MODEL SAVED!"
            )

        print("=" * 60)


    # ========================================================
    # Finished
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Best model:",
        best_checkpoint
    )

    print(
        "Latest model:",
        latest_checkpoint
    )

    print(
        "Best PSNR:",
        best_psnr
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()