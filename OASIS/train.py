import os, time, json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import OASISConfig
from .datasets import OASISDataset
from .model import UNetSmall
from .losses import SegRegLoss
from .utils import split_indices_by_kind

# ---------------- DDP helpers ----------------

def _ddp_on():
    return dist.is_available() and dist.is_initialized()

def _rank():
    return dist.get_rank() if _ddp_on() else 0

def _world():
    return dist.get_world_size() if _ddp_on() else 1

def _is_main():
    return _rank() == 0

def setup_ddp_from_env() -> bool:
    """
    torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK.
    """
    if "RANK" not in os.environ:
        return False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    dist.barrier()
    return True

def cleanup_ddp():
    if _ddp_on():
        dist.barrier()
        dist.destroy_process_group()

def _reduce_mean_scalar(x: float, device: torch.device) -> float:
    t = torch.tensor([x], device=device, dtype=torch.float32)
    if _world() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= _world()
    return float(t.item())

def _broadcast_int_list(lst, device: torch.device):
    """
    Broadcast list[int] from rank0 -> all ranks.
    """
    if not _ddp_on():
        return lst
    if _is_main():
        t = torch.tensor(lst, device=device, dtype=torch.int64)
        n = torch.tensor([t.numel()], device=device, dtype=torch.int64)
    else:
        n = torch.tensor([0], device=device, dtype=torch.int64)

    dist.broadcast(n, src=0)
    if not _is_main():
        t = torch.empty((int(n.item()),), device=device, dtype=torch.int64)
    dist.broadcast(t, src=0)
    return t.tolist()

# ---------------- training ----------------

def train(cfg: OASISConfig):
    ddp = setup_ddp_from_env()

    use_cuda = torch.cuda.is_available() and cfg.device == "cuda"
    device = torch.device("cuda", torch.cuda.current_device()) if use_cuda else torch.device("cpu")

    # Tensor-core friendly defaults
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if _is_main():
        print(f"DDP={ddp} world={_world()} device={device}")

    # --- dataset + deterministic split (rank0 -> broadcast) ---
    ds_full = OASISDataset(
        root_dir=cfg.data_root,
        split="train",
        kinds=cfg.kinds,
        hw=cfg.hw,
        file_pattern=cfg.file_pattern
    )

    if ddp:
        if _is_main():
            tr_idx, va_idx = split_indices_by_kind(
                ds_full,
                n_train_per_kind=cfg.n_train_per_kind,
                n_val_per_kind=cfg.n_val_per_kind,
                seed=cfg.seed
            )
        else:
            tr_idx, va_idx = [], []
        tr_idx = _broadcast_int_list(tr_idx, device)
        va_idx = _broadcast_int_list(va_idx, device)
    else:
        tr_idx, va_idx = split_indices_by_kind(
            ds_full,
            n_train_per_kind=cfg.n_train_per_kind,
            n_val_per_kind=cfg.n_val_per_kind,
            seed=cfg.seed
        )

    ds_train = Subset(ds_full, tr_idx)
    ds_val   = Subset(ds_full, va_idx)

    if _is_main():
        print(f"Train: {len(ds_train)} | Val: {len(ds_val)}")

    # --- DDP samplers ---
    train_sampler = DistributedSampler(ds_train, shuffle=True, seed=cfg.seed) if ddp else None
    val_sampler   = DistributedSampler(ds_val, shuffle=False) if ddp else None

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    model = UNetSmall(base=cfg.base_channels).to(device)
    lossf = SegRegLoss(
        alpha_reg=cfg.alpha_reg, alpha_tv=cfg.alpha_tv,
        w_ch=(cfg.w_ER, cfg.w_NR),
        w_regions=(cfg.W_ER, cfg.W_NR, cfg.W_overlap)
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if _is_main():
        os.makedirs(cfg.model_dir, exist_ok=True)
        # Save run config for provenance
        with open(os.path.join(cfg.model_dir, "config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2, default=str)
    if ddp:
        dist.barrier()

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # --- TRAIN ---
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "reg": 0.0, "tv": 0.0}

        for I_in, t, valid, aux in dl_train:
            I_in  = I_in.to(device, non_blocking=True).float()
            t     = t.to(device, non_blocking=True).float()
            valid = valid.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                I_hat = model(I_in)
                d = lossf(I_hat, I_in, t=t, valid=valid)
                loss = d["loss"]

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            for k in sums:
                sums[k] += float(d[k])

        ntr = max(1, len(dl_train))
        train_avg_local = {k: sums[k] / ntr for k in sums}
        train_avg = {k: _reduce_mean_scalar(train_avg_local[k], device) for k in sums}

        if _is_main():
            msg = " | ".join([f"{k}:{train_avg[k]:.4f}" for k in sums])
            print(f"Epoch {epoch:03d} train: {msg} | time {time.time()-t0:.1f}s")

        # --- VAL ---
        model.eval()
        vsums = {"loss": 0.0, "reg": 0.0, "tv": 0.0}
        with torch.no_grad():
            for I_in, t, valid, aux in dl_val:
                I_in  = I_in.to(device, non_blocking=True).float()
                t     = t.to(device, non_blocking=True).float()
                valid = valid.to(device, non_blocking=True).float()

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                    I_hat = model(I_in)
                    d = lossf(I_hat, I_in, t=t, valid=valid)

                for k in vsums:
                    vsums[k] += float(d[k])

        nva = max(1, len(dl_val))
        val_avg_local = {k: vsums[k] / nva for k in vsums}
        val_avg = {k: _reduce_mean_scalar(val_avg_local[k], device) for k in vsums}
        val_loss = val_avg["loss"]

        if _is_main():
            vmsg = " | ".join([f"{k}:{val_avg[k]:.4f}" for k in vsums])
            print(f"Epoch {epoch:03d}  val : {vmsg}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
                ckpt = os.path.join(cfg.model_dir, cfg.best_training_weights)
                state = model.module.state_dict() if ddp else model.state_dict()
                torch.save(state, ckpt)
                print(f"  ✓ Saved best to {ckpt} (val {best_val:.4f})")
            else:
                no_improve += 1
                print(f"  no improvement for {no_improve}/{cfg.patience} epochs")

        # broadcast stop decision so all ranks exit together
        if ddp:
            stop = torch.tensor([0], device=device, dtype=torch.int64)
            if _is_main() and (no_improve >= cfg.patience):
                stop[0] = 1
            dist.broadcast(stop, src=0)
            if stop.item() == 1:
                if _is_main():
                    print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val {best_val:.4f})")
                break
        else:
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val {best_val:.4f})")
                break

    if _is_main():
        last_ckpt = os.path.join(cfg.model_dir, cfg.last_training_weights)
        state = model.module.state_dict() if ddp else model.state_dict()
        torch.save(state, last_ckpt)
        print(f"Done. Saved last to {last_ckpt}")

    cleanup_ddp()
    return model
