"""Utilidades para fijar la semilla aleatoria."""

import random
import numpy as np
import torch

def set_seed(seed_value: int):
    """Fija la semilla para reproducibilidad."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
