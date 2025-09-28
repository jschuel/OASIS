import os, time, torch
from torch.utils.data import DataLoader, Subset
from .config import OASISConfig
from .datasets import OASISDataset
from .model import UNetSmall
from .losses import SegRegLoss
from .utils import split_indices_by_kind

def train(cfg: OASISConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")
    print(f"Using device: {device}")

    ds_full = OASISDataset(root_dir=cfg.data_root, split="train", kinds=cfg.kinds, hw=cfg.hw, file_pattern=cfg.file_pattern)
    tr_idx, va_idx = split_indices_by_kind(ds_full,
                                           n_train_per_kind=cfg.n_train_per_kind,
                                           n_val_per_kind=cfg.n_val_per_kind,
                                           seed=cfg.seed)
    ds_train = Subset(ds_full, tr_idx)
    ds_val   = Subset(ds_full, va_idx)

    print(f"Train: {len(ds_train)} | Val: {len(ds_val)}")
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    model = UNetSmall(base=cfg.base_channels).to(device)
    lossf = SegRegLoss(
        alpha_reg=cfg.alpha_reg, alpha_tv=cfg.lam_tv,
        w_ch=(cfg.w_ER, cfg.w_NR), w_regions=(cfg.W_ER, cfg.W_NR, cfg.W_overlap)
    ).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    os.makedirs(cfg.model_dir, exist_ok=True)
    best_val = float('inf'); best_epoch = 0; no_improve = 0

    for epoch in range(1, cfg.epochs+1):
        # --- TRAIN ---
        model.train()
        t0 = time.time()
        logs = {"loss":0.0, "reg":0.0, "tv":0.0}

        for batch in dl_train:
            I_in, t, valid, aux = batch
            I_in  = I_in.to(device, non_blocking=True).float()
            t     = t.to(device, non_blocking=True).float()
            valid = valid.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                I_hat = model(I_in)
                d = lossf(I_hat, I_in, t=t, valid=valid)
                loss = d["loss"]

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            for k in logs: logs[k] += float(d[k])

        ntr = max(1, len(dl_train))
        train_msg = " | ".join([f"{k}:{logs[k]/ntr:.4f}" for k in logs])
        print(f"Epoch {epoch:03d} train: {train_msg} | time {time.time()-t0:.1f}s")

        # --- VAL ---
        model.eval()
        vlogs = {"loss":0.0, "reg":0.0, "tv":0.0}
        with torch.no_grad():
            for batch in dl_val:
                I_in, t, valid, aux = batch
                I_in  = I_in.to(device, non_blocking=True).float()
                t     = t.to(device, non_blocking=True).float()
                valid = valid.to(device, non_blocking=True).float()

                I_hat = model(I_in)
                d = lossf(I_hat, I_in, t=t, valid=valid)
                for k in vlogs: vlogs[k] += float(d[k])

        nva = max(1, len(dl_val))
        val_msg  = " | ".join([f"{k}:{vlogs[k]/nva:.4f}" for k in vlogs])
        val_loss = vlogs["loss"]/nva
        print(f"Epoch {epoch:03d}  val : {val_msg}")

        # --- Early stopping / checkpoint ---
        if val_loss < best_val - 1e-6:
            best_val = val_loss; best_epoch = epoch; no_improve = 0
            ckpt = os.path.join(cfg.model_dir, cfg.best_training_weights)
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved best to {ckpt} (val {best_val:.4f})")
        else:
            no_improve += 1
            print(f"  no improvement for {no_improve}/{cfg.patience} epochs")
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val {best_val:.4f})")
                break

    last_ckpt = os.path.join(cfg.model_dir, cfg.last_training_weights)
    torch.save(model.state_dict(), last_ckpt)
    print(f"Done. Saved last to {last_ckpt}")
    return model
