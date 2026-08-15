import numpy as np
from skimage.metrics import structural_similarity

def psnr(pred, target, data_range=None):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = np.mean((pred-target)**2)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = float(target.max() - target.min())
        if data_range <= 0:
            data_range = 1.0
    return float(10*np.log10((data_range**2)/mse))

def ssim(pred, target, data_range=None):
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.ndim == 3:
        pred = pred.squeeze()
    if target.ndim == 3:
        target = target.squeeze()
    if data_range is None:
        data_range = float(target.max() - target.min())
        if data_range <= 0:
            data_range = 1.0
    return float(structural_similarity(pred, target, data_range=data_range))
