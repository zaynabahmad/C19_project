"""Plain NumPy/OpenCV image helpers for the data pipeline (no PyTorch needed, so they are easy to test).

Images are uint8 arrays (H, W, 3), RGB. Pairs are always cropped/flipped TOGETHER so that the
blurred patch and the sharp patch stay perfectly aligned.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np


def load_rgb_uint8(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_split(json_path, split: str):
    """List of (blur_path, sharp_path) for split in {'train', 'val', 'test'} from the subset JSON."""
    cfg = json.load(open(json_path))
    root = Path(cfg["root"])
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Download the data and re-run: python -m src.make_subset")
    return [(root / p["blur"], root / p["sharp"]) for p in cfg[split]]


def random_crop_pair(a: np.ndarray, b: np.ndarray, size: int, rnd=random):
    """Same random size x size window from both images."""
    assert a.shape == b.shape, f"blur/sharp shapes differ: {a.shape} vs {b.shape}"
    h, w = a.shape[:2]
    assert size <= h and size <= w, f"patch size {size} larger than image {h}x{w}"
    y, x = rnd.randint(0, h - size), rnd.randint(0, w - size)
    return a[y:y + size, x:x + size], b[y:y + size, x:x + size]


def center_crop_pair(a: np.ndarray, b: np.ndarray, size: int):
    assert a.shape == b.shape
    h, w = a.shape[:2]
    assert size <= h and size <= w
    y, x = (h - size) // 2, (w - size) // 2
    return a[y:y + size, x:x + size], b[y:y + size, x:x + size]


def augment_pair(a: np.ndarray, b: np.ndarray, rnd=random):
    """Random horizontal and vertical flip, applied identically to both images."""
    if rnd.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    if rnd.random() < 0.5:
        a, b = a[::-1], b[::-1]
    return np.ascontiguousarray(a), np.ascontiguousarray(b)