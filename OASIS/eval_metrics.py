import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import OASISConfig
from .datasets import OASISDataset
from .model import UNetSmall
from .utils import invert_scale_log10p_torch

@torch.no_grad()
def eval_bias_variance_4to5keV(
    cfg: OASISConfig,
    test_df,
    model_path: str,
    species: str = "hybrid",
    eps: float = 1e-12,
    num_workers: int = 2,
):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    ds = OASISDataset(
        root_dir=cfg.data_root,
        split=cfg.eval_split,
        kinds=(species,),
        hw=cfg.hw,
        file_pattern=cfg.file_pattern
    )

    e = np.asarray(test_df["ER_truth_energy"].values, dtype=np.float64)
    sel = np.where((e >= 4.0) & (e < 5.0))[0]
    if sel.size == 0:
        raise ValueError("No events satisfy 4.0 <= ER_truth_energy < 5.0")

    #df index aligns with ds index
    ds_sel = Subset(ds, sel.tolist())

    dl = DataLoader(
        ds_sel,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = UNetSmall(base=cfg.base_channels).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    rel_all = []

    for I_in, t, valid, aux in dl:
        I_in  = I_in.to(device).float()
        t     = t.to(device).float()
        valid = valid.to(device).float()

        # aux collates into dict of tensors [B]; make [B,1,1,1]
        s      = aux["s"].to(device).float().view(-1, 1, 1, 1)
        ymax10 = aux["ymax10"].to(device).float().view(-1, 1, 1, 1)

        I_hat = model(I_in)

        # ER channel only
        pred_raw = invert_scale_log10p_torch(I_hat[:, 0:1], s, ymax10) * valid
        true_raw = invert_scale_log10p_torch(t[:, 0:1],     s, ymax10) * valid

        Ip = pred_raw.sum(dim=(1,2,3))
        It = true_raw.sum(dim=(1,2,3)).clamp_min(eps)

        rel = ((Ip - It) / It * 100.0).detach().cpu().numpy()
        rel_all.append(rel)

    rel_all = np.concatenate(rel_all, axis=0)

    out = {
        "N_4to5": int(rel_all.size),
        "median_pct": float(np.median(rel_all)),
        "p25_pct": float(np.percentile(rel_all, 25)),
        "p75_pct": float(np.percentile(rel_all, 75)),
    }
    print(f"[4-5 keV] N={out['N_4to5']} | median={out['median_pct']:.3f}% | "
          f"p25={out['p25_pct']:.3f}% | p75={out['p75_pct']:.3f}%")
    return out
