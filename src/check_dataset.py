
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import GoProDeblur
from src.utils import SEED, set_seed


def main():
    set_seed()
    train = GoProDeblur("train", patch_size=128)
    val = GoProDeblur("val", patch_size=256)
    full = GoProDeblur("test")  # full images
    print(f"train {len(train)} | val {len(val)} | test {len(full)} pairs")
    b, s = train[0]
    print("train item:", tuple(b.shape), b.dtype, f"range [{b.min():.2f}, {b.max():.2f}]")
    b, s = full[0]
    print("test item (full image):", tuple(b.shape))
    assert train[0][0].shape == (3, 128, 128) and val[0][0].shape == (3, 256, 256)

    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=2, generator=g)
    blur, sharp = next(iter(loader))
    print("batch:", tuple(blur.shape), tuple(sharp.shape))

    def row(t):
        return np.concatenate([(x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for x in t], axis=1)

    grid = np.concatenate([row(blur), row(sharp)], axis=0)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("results/figures/patch_check.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
    