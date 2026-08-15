import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def load_npy(path):
    x = np.load(path)
    if not np.isfinite(x).all():
        raise ValueError(f"NaN/Inf found in {path}")
    x = x.astype(np.float32, copy=False)
    if x.ndim == 2:
        x = x[None, ...]
    elif x.ndim == 3:
        # Accept C,H,W or H,W,C; prefer single-channel interpretation.
        if x.shape[0] in (1, 3):
            pass
        elif x.shape[-1] in (1, 3):
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported shape {x.shape} in {path}")
    else:
        raise ValueError(f"Unsupported array ndim {x.ndim} in {path}")
    return torch.from_numpy(x)

def pair_files(noisy_dir, gt_dir, mode="same_stem"):
    noisy = sorted(Path(noisy_dir).glob("*.npy"))
    gt = sorted(Path(gt_dir).glob("*.npy"))
    if mode == "same_stem":
        gt_map = {p.stem: p for p in gt}
        pairs = [(p, gt_map[p.stem]) for p in noisy if p.stem in gt_map]
        if len(pairs) != len(noisy) or len(pairs) != len(gt):
            missing_n = [p.name for p in noisy if p.stem not in gt_map][:10]
            noisy_stems = {p.stem for p in noisy}
            missing_g = [p.name for p in gt if p.stem not in noisy_stems][:10]
            raise RuntimeError(
                "Safe same-stem pairing failed. "
                f"Noisy={len(noisy)}, GT={len(gt)}, pairs={len(pairs)}. "
                f"Example unmatched noisy={missing_n}, GT={missing_g}. "
                "Do not train until the official pairing rule is known."
            )
        return pairs
    if mode == "sorted":
        if len(noisy) != len(gt):
            raise RuntimeError("Sorted pairing requested but file counts differ.")
        return list(zip(noisy, gt))
    raise ValueError(f"Unknown pairing mode: {mode}")

class NpyPairedDataset(Dataset):
    def __init__(self, pairs, patch_size=None, train=True):
        self.pairs = pairs
        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def _crop(self, x, y):
        if not self.patch_size:
            return x, y
        h, w = x.shape[-2:]
        ph = min(self.patch_size, h)
        pw = min(self.patch_size, w)
        if h == ph:
            top = 0
        else:
            top = np.random.randint(0, h - ph + 1)
        if w == pw:
            left = 0
        else:
            left = np.random.randint(0, w - pw + 1)
        # If target is larger, crop using the inferred integer scale.
        th, tw = y.shape[-2:]
        sx = th / h
        sy = tw / w
        if abs(sx - sy) > 1e-6:
            raise ValueError(f"Non-uniform scale: input {x.shape}, target {y.shape}")
        scale = sx
        top_y = int(round(top * scale))
        left_y = int(round(left * scale))
        ph_y = min(int(round(ph * scale)), th - top_y)
        pw_y = min(int(round(pw * scale)), tw - left_y)
        return x[..., top:top+ph, left:left+pw], y[..., top_y:top_y+ph_y, left_y:left_y+pw_y]

    def _aug(self, x, y):
        if not self.train:
            return x, y
        if np.random.rand() < 0.5:
            x = torch.flip(x, [-1]); y = torch.flip(y, [-1])
        if np.random.rand() < 0.5:
            x = torch.flip(x, [-2]); y = torch.flip(y, [-2])
        k = np.random.randint(0, 4)
        if k:
            x = torch.rot90(x, k, [-2, -1]); y = torch.rot90(y, k, [-2, -1])
        return x, y

    def __getitem__(self, idx):
        npy, gt = self.pairs[idx]
        x = load_npy(npy)
        y = load_npy(gt)
        if x.shape[0] != 1 or y.shape[0] != 1:
            raise ValueError(f"Expected grayscale arrays, got {x.shape}, {y.shape}")
        x, y = self._crop(x, y)
        x, y = self._aug(x, y)
        return x, y, str(npy), str(gt)
