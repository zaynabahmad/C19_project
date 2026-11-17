"""Shared helpers. Keeps the random seed in ONE place (course requirement:
a fixed random seed for the headline result, Instr. Sec. 9.3)."""
import random

import numpy as np

SEED = 42  # TEAM DECISION PENDING: confirm or change; then never change it silently.


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch. Full GPU determinism is not enforced."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # PyTorch is optional for the classical-baseline scripts
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
