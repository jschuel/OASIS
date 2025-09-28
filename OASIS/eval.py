
import os, torch, numpy as np
from torch.utils.data import DataLoader
from .config import OASISConfig
from .datasets import OASISDataset
from .model import UNetSmall
from .losses import SegRegLoss
from .utils import ensure_bchw, invert_scale_log10p_torch

@torch.no_grad()
def evaluate(cfg: OASISConfig, model_path: str = None, species: str = "NR"):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")
    model_path = model_path or (os.path.join(cfg.model_dir, cfg.model_file) if cfg.model_file is None else cfg.model_file)

    ds = OASISDataset(root_dir=cfg.data_root, split=cfg.eval_split, kinds=(species,), hw=cfg.hw, file_pattern=cfg.file_pattern)
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=1, pin_memory=True)

    model = UNetSmall(base=cfg.base_channels).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    lossf = SegRegLoss(
        alpha_reg=cfg.alpha_reg, alpha_tv=cfg.alpha_tv,
        w_ch=(cfg.w_ER, cfg.w_NR), w_regions=(cfg.W_ER, cfg.W_NR, cfg.W_overlap)
    ).to(device)

    agg = {"loss":0.0, "reg":0.0, "tv":0.0}
    n_batches = 0

    for batch in dl:
        I_in, t, valid, aux = batch
        I_in = I_in.to(device).float()
        t    = t.to(device).float()
        valid= valid.to(device).float()

        I_hat = model(I_in)
        d = lossf(I_hat, I_in, t=t, valid=valid)
        for k in agg: agg[k] += float(d[k])
        n_batches += 1

    for k in agg: agg[k] /= max(n_batches,1)
    print("[EVAL] " + " | ".join([f"{k}:{agg[k]:.4f}" for k in ["loss","reg","tv"]]))
    return model, ds, agg

@torch.no_grad()
def view_and_process(model, dataset, idx=0, device=None, invert=False, mask=True, plot=True, return_NR=False, save_path=None):
    import matplotlib.pyplot as plt
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    sample = dataset[idx]
    I_in, t, valid, aux = sample
    s = torch.tensor(aux.get("s", 1.0)).view(1,1,1,1)
    ymax10 = torch.tensor(aux.get("ymax10", 1.0)).view(1,1,1,1)

    I_in   = ensure_bchw(I_in.float().to(device))
    t      = ensure_bchw(t.float().to(device))
    valid  = ensure_bchw(valid.float().to(device))
    s      = s.to(device).type_as(I_in)
    ymax10 = ymax10.to(device).type_as(I_in)

    model.eval()
    I_hat = model(I_in)

    if invert:
        I_hat_ER_raw = invert_scale_log10p_torch(I_hat[:,0:1], s, ymax10)
        I_hat_NR_raw = invert_scale_log10p_torch(I_hat[:,1:2], s, ymax10)
        t_ER_raw     = invert_scale_log10p_torch(t[:,0:1],     s, ymax10)
        t_NR_raw     = invert_scale_log10p_torch(t[:,1:2],     s, ymax10)
        if mask:
            I_hat_ER_raw *= valid; I_hat_NR_raw *= valid; t_ER_raw *= valid; t_NR_raw *= valid
        to_show = {
            "Input (scaled)": I_in[0,0].cpu(),
            "Pred ER (raw)": I_hat_ER_raw[0,0].cpu(),
            "GT ER (raw)": t_ER_raw[0,0].cpu(),
            "Pred NR (raw)": I_hat_NR_raw[0,0].cpu(),
            "GT NR (raw)": t_NR_raw[0,0].cpu(),
            "Residual ER (raw)": (I_hat_ER_raw - t_ER_raw)[0,0].cpu().abs(),
            "Residual NR (raw)": (I_hat_NR_raw - t_NR_raw)[0,0].cpu().abs(),
        }
    else:
        to_show = {
            "Input (scaled)": I_in[0,0].cpu(),
            "Pred ER (scaled)": I_hat[0,0].cpu(),
            "GT ER (scaled)": t[0,0].cpu(),
            "Pred NR (scaled)": I_hat[0,1].cpu(),
            "GT NR (scaled)": t[0,1].cpu(),
            "Residual ER (scaled)": (I_hat[0,0]-t[0,0]).cpu().abs(),
            "Residual NR (scaled)": (I_hat[0,1]-t[0,1]).cpu().abs(),
        }

    if plot:
        n = len(to_show)
        cols = 3
        rows = (n+cols-1)//cols
        plt.figure(figsize=(4*cols, 3.5*rows))
        for i,(k,v) in enumerate(to_show.items(), 1):
            ax = plt.subplot(rows, cols, i)
            im = ax.imshow(v.numpy())
            ax.set_title(k); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=160)
            print(f"Saved figure to {save_path}")
        else:
            plt.show()

    # return scalars for quick sanity checks
    if invert:
        ERp = I_hat_ER_raw[0,0].cpu().numpy().sum(); ERt = t_ER_raw[0,0].cpu().numpy().sum()
        NRp = I_hat_NR_raw[0,0].cpu().numpy().sum(); NRt = t_NR_raw[0,0].cpu().numpy().sum()
        ERrel = (ERp-ERt)/ERt*100.0
        NRrel = (NRp-NRt)/NRt*100.0
        return (ERp, ERt, ERrel) if not return_NR else (ERp, ERt, ERrel, NRp, NRt, NRrel)
    return {}
