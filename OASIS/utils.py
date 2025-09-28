import torch
import numpy as np
import random
from typing import Sequence, Tuple, List
from torch.utils.data import Subset
from .datasets import OASISDataset

def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x

def invert_scale_log10p_torch(y: torch.Tensor, s: torch.Tensor, ymax10: torch.Tensor) -> torch.Tensor:
    """
    y in [0,1], s,ymax10 are broadcastable (e.g., [B,1,1,1]).
    """
    return s * (torch.pow(10.0, y * ymax10) - 1.0)

def scale_image_log10p(x: np.ndarray, pclip: float = 99.0, thresh: float = 0.0):
    """
    Numpy scaling to match your notebook:
      - zero out below thresh
      - s = percentile(x, pclip)
      - y = log10(1 + x/s), then normalize by ymax10 = y.max()
    Returns: y_norm [0,1], s, ymax10
    """
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float32)
    if thresh < 0:
        raise ValueError("thresh must be >= 0")
    x = x.copy()
    x[x < thresh] = 0
    s = float(np.percentile(x, pclip))
    s = max(s, 1e-12)
    y = np.log10(1.0 + x / s)
    ymax10 = float(max(y.max(), 1e-12))
    y_norm = (y / ymax10).astype(np.float32)
    return y_norm, s, ymax10

def split_indices_by_kind(ds: OASISDataset,
                          n_train_per_kind=18000,
                          n_val_per_kind=2000,
                          seed=1337) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_kind = {}
    for i, k in enumerate(ds.labels):
        by_kind.setdefault(k, []).append(i)

    train_idx, val_idx = [], []
    for k, idxs in by_kind.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        need = n_train_per_kind + n_val_per_kind
        if len(idxs) < need:
            raise ValueError(f"Not enough samples for kind '{k}': have {len(idxs)}, need {need}.")
        train_idx.extend(idxs[:n_train_per_kind])
        val_idx.extend(idxs[n_train_per_kind:n_train_per_kind+n_val_per_kind])

    return sorted(train_idx), sorted(val_idx)
