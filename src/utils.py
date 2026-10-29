"""Shared helpers. Keeps the random seed in ONE place (course requirement:
a fixed random seed for the headline result, Instr. Sec. 9.3)."""
import random

import numpy as np

SEED = 42  # TEAM DECISION PENDING: confirm or change; then never change it silently.


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy. (PyTorch seeding is added at M2.)"""
    random.seed(seed)
    np.random.seed(seed)
