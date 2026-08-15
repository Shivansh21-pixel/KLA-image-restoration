import argparse, json
from pathlib import Path
import numpy as np
import yaml

def stats(paths, limit=None):
    paths = list(paths)
    if limit:
        sample = paths[:limit]
    else:
        sample = paths
    shapes, dtypes = {}, {}
    mins, maxs, means, stds = [], [], [], []
    bad = []
    for p in sample:
        try:
            a = np.load(p)
            if not np.isfinite(a).all():
                bad.append(str(p))
                continue
            shapes[str(a.shape)] = shapes.get(str(a.shape), 0) + 1
            dtypes[str(a.dtype)] = dtypes.get(str(a.dtype), 0) + 1
            mins.append(float(a.min())); maxs.append(float(a.max()))
            means.append(float(a.mean())); stds.append(float(a.std()))
        except Exception as e:
            bad.append(f"{p}: {e}")
    return {
        "files": len(paths),
        "inspected": len(sample),
        "shapes": shapes,
        "dtypes": dtypes,
        "min": min(mins) if mins else None,
        "max": max(maxs) if maxs else None,
        "mean_of_means": float(np.mean(means)) if means else None,
        "mean_of_stds": float(np.mean(stds)) if stds else None,
        "bad_files": bad[:20],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    train = Path(cfg["train_root"])
    test = Path(cfg["test_root"])
    noisy = train / cfg["noisy_dir_name"]
    gt = train / cfg["gt_dir_name"]
    test_noisy = test / cfg["noisy_dir_name"]
    print("Noisy train:", noisy)
    print("GT train:", gt)
    print("Test noisy:", test_noisy)
    report = {
        "train_noisy": stats(sorted(noisy.glob("*.npy"))),
        "train_gt": stats(sorted(gt.glob("*.npy"))),
        "test_noisy": stats(sorted(test_noisy.glob("*.npy"))),
        "pairing_mode": cfg.get("pairing"),
    }
    Path("reports").mkdir(exist_ok=True)
    json.dump(report, open("reports/dataset_report.json","w",encoding="utf-8"), indent=2)
    print(json.dumps(report, indent=2))
    print("\nNOTE: Pairing is NOT changed or assumed by this inspection script.")

if __name__ == "__main__":
    main()
