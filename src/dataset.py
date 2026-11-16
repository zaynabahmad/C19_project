"""PyTorch Dataset over the fixed GoPro subset lists written by src/make_subset.py.

  train: random patch_size x patch_size crop (same window for blur and sharp) + random flips
  val/test: center crop of patch_size if given, otherwise the FULL image
Each item is a pair of float32 tensors (3, H, W) with values in [0, 1]: (blurred, sharp).
"""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.image_ops import augment_pair, center_crop_pair, load_rgb_uint8, random_crop_pair, read_split


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


class GoProDeblur(Dataset):
    def __init__(self, split: str, patch_size=None, json_path="data/subsets/gopro_subset.json"):
        assert split in ("train", "val", "test")
        self.split, self.patch_size = split, patch_size
        self.pairs = read_split(json_path, split)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        blur, sharp = (load_rgb_uint8(p) for p in self.pairs[i])
        if self.split == "train":
            if self.patch_size:
                blur, sharp = random_crop_pair(blur, sharp, self.patch_size, random)
            blur, sharp = augment_pair(blur, sharp, random)
        elif self.patch_size:
            blur, sharp = center_crop_pair(blur, sharp, self.patch_size)
        return to_tensor(blur), to_tensor(sharp)