"""Image-quality metrics used for ALL methods (classical baseline and the model).

Images are float arrays in [0, 1], shape (H, W, 3), RGB. Both metrics are computed on the full
image with scikit-image, data_range=1.0. SSIM uses scikit-image's default settings
(7x7 uniform window, channel_axis=2). Other papers may use different SSIM settings, so
numbers here are comparable ACROSS OUR OWN methods, not automatically with published tables.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, test, data_range=1.0))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    return float(structural_similarity(ref, test, channel_axis=2, data_range=1.0))
