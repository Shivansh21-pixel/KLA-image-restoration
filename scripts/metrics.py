import torch
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure


def calculate_metrics(pred, target):
    """
    pred   = model ka output
    target = GT / clean image
    """

    # Metrics ke liye values ko 0-1 range me rakho.
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    psnr = peak_signal_noise_ratio(
        pred,
        target,
        data_range=1.0
    )

    ssim = structural_similarity_index_measure(
        pred,
        target,
        data_range=1.0
    )

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
    }