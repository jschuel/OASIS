import os, json, argparse
from dataclasses import replace

from .config import OASISConfig
from .train import train

def apply_ratios(cfg: OASISConfig, rc: float, rn: float, ro: float) -> OASISConfig:
    # Baselines
    wNR = 1.0
    WER = 1.0
    return replace(
        cfg,
        w_NR=wNR,
        w_ER=rc * wNR,
        W_ER=WER,
        W_NR=rn * WER,
        W_overlap=ro * WER,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rc", type=float, required=True)
    p.add_argument("--rn", type=float, required=True)
    p.add_argument("--ro", type=float, required=True)
    p.add_argument("--run_id", type=str, required=True)

    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--model_root", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    args = p.parse_args()
    cfg = OASISConfig()

    # generic overrides
    if args.data_root is not None:
        cfg = replace(cfg, data_root=args.data_root)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.epochs is not None:
        cfg = replace(cfg, epochs=args.epochs)
    if args.patience is not None:
        cfg = replace(cfg, patience=args.patience)
    if args.batch_size is not None:
        cfg = replace(cfg, batch_size=args.batch_size)
    if args.lr is not None:
        cfg = replace(cfg, lr=args.lr)

    cfg = apply_ratios(cfg, args.rc, args.rn, args.ro)

    model_root = args.model_root or cfg.model_dir
    run_dir = os.path.join(model_root, args.run_id)
    cfg = replace(
        cfg,
        model_dir=run_dir,
        best_training_weights="best.pt",
        last_training_weights="last.pt",
        # for convenience: evaluation will default to best.pt in this run dir
        model_file=os.path.join(run_dir, "best.pt"),
    )

    # rank0 will also write config.json from train()
    if "RANK" not in os.environ or os.environ.get("RANK") == "0":
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "ratios.json"), "w") as f:
            json.dump(
                dict(run_id=args.run_id, rc=args.rc, rn=args.rn, ro=args.ro,
                     w_ER=cfg.w_ER, w_NR=cfg.w_NR, W_ER=cfg.W_ER, W_NR=cfg.W_NR, W_overlap=cfg.W_overlap),
                f, indent=2
            )
        print(f"[run_id={args.run_id}] rc={args.rc} rn={args.rn} ro={args.ro}")
        print(f"  w_ER={cfg.w_ER} w_NR={cfg.w_NR} | W_ER={cfg.W_ER} W_NR={cfg.W_NR} W_ov={cfg.W_overlap}")
        print(f"  model_dir={cfg.model_dir}")

    train(cfg)

if __name__ == "__main__":
    main()
