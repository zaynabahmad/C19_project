"""Train the compact Transformer on the fixed GoPro training subset.   Run from the repo root.

  Smoke test (20 steps, writes to *_smoke files):
      python3 -m src.train --model tiny --epochs 1 --max-steps 20 --tag smoke
  Real run:
      python3 -m src.train --model small --epochs 30

Training: random PATCH x PATCH crops, random flips, L1 loss, AdamW, cosine learning-rate decay.
Validation (every epoch): PSNR on 256x256 center crops of the 56 VALIDATION pairs, computed in torch.
(The final test numbers are computed separately with src/metrics.py on the full test images.)
The best checkpoint (by validation PSNR) and the last one are saved to checkpoints/ (git-ignored).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from src.dataset import GoProDeblur
from src.model import CONFIGS, build_model, count_parameters
from src.utils import SEED, set_seed


def psnr_per_image(pred, target):
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


@torch.no_grad()
def validate(model, loader, device):
    """Mean PSNR of the model output and of the untouched blurred input, over the validation set."""
    model.eval()
    out_psnr, in_psnr = [], []
    for blur, sharp in loader:
        blur, sharp = blur.to(device), sharp.to(device)
        out_psnr.append(psnr_per_image(model(blur).clamp(0, 1), sharp).cpu())
        in_psnr.append(psnr_per_image(blur, sharp).cpu())
    model.train()
    return torch.cat(out_psnr).mean().item(), torch.cat(in_psnr).mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(CONFIGS), default="small")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=4, help="random patches drawn per training image per epoch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default=None, help="name used for output files (default: model name)")
    ap.add_argument("--resume", default=None, help="continue from a *_last.pth checkpoint (same arguments as the original run)")
    args = ap.parse_args()
    tag = args.tag or args.model

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU available to PyTorch, training on the CPU will be very slow.")
    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"model '{args.model}': {n_params:,} parameters ({n_params / 1e6:.3f} M) on {device}")

    train_ds = ConcatDataset([GoProDeblur("train", patch_size=args.patch)] * args.repeat)
    val_ds = GoProDeblur("val", patch_size=args.val_crop)
    gen = torch.Generator().manual_seed(args.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                              num_workers=args.workers, generator=gen, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.workers)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)
    print(f"{len(train_ds)} patches/epoch, {steps_per_epoch} steps/epoch, {total_steps} steps in total")

    ckpt_dir, table_dir = Path("checkpoints"), Path("results/tables")
    ckpt_dir.mkdir(exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    def save(path, epoch, val_psnr):
        torch.save({"model": model.state_dict(), "model_name": args.model, "epoch": epoch,
                    "val_psnr": val_psnr, "params": n_params, "args": vars(args),
                    # state needed by --resume
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "rows": rows, "step": step,
                    "best": best, "train_secs": train_secs, "elapsed": time.time() - t0}, path)

    val0, in0 = validate(model, val_loader, device)
    print(f"before training: validation PSNR  blurred input = {in0:.2f} dB, untrained model = {val0:.2f} dB")

    rows, step, best, t0, done = [], 0, float("-inf"), time.time(), False
    train_secs = 0.0  # time spent in training steps only (validation excluded)
    start_epoch = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rows, step, best, train_secs = ck["rows"], ck["step"], ck["best"], ck["train_secs"]
        t0 = time.time() - ck["elapsed"]
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {args.resume}: epoch {ck['epoch']} done, continuing at epoch {start_epoch} (step {step})")
    for epoch in range(start_epoch, args.epochs + 1):
        run_loss, n = 0.0, 0
        t_ep = time.time()
        for blur, sharp in train_loader:
            blur, sharp = blur.to(device, non_blocking=True), sharp.to(device, non_blocking=True)
            loss = F.l1_loss(model(blur), sharp)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is {loss.item()} at step {step}: training diverged")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss, n = run_loss + loss.item(), n + 1
            if step >= total_steps:
                done = True
                break
        train_secs += time.time() - t_ep
        val_psnr, _ = validate(model, val_loader, device)
        secs = time.time() - t0
        rows.append({"epoch": epoch, "step": step, "train_l1": run_loss / n, "val_psnr": val_psnr,
                     "lr": sched.get_last_lr()[0], "seconds": round(secs),
                     "train_s_per_step": round(train_secs / step, 3)})
        with open(table_dir / f"train_log_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if val_psnr > best:
            best = val_psnr
            save(ckpt_dir / f"{tag}_best.pth", epoch, val_psnr)
        save(ckpt_dir / f"{tag}_last.pth", epoch, val_psnr)
        print(f"epoch {epoch:3d} | step {step:5d} | L1 {run_loss / n:.4f} | val PSNR {val_psnr:.2f} dB "
              f"(best {best:.2f}) | {secs / 60:.1f} min total | {train_secs / step:.2f} s/step (training only)")
        if done:
            break

    summary = {"model": args.model, "tag": tag, "parameters": n_params, "steps": step,
               "minutes": round((time.time() - t0) / 60, 2), "train_s_per_step": round(train_secs / max(step, 1), 3),
               "best_val_psnr": best,
               "val_psnr_blurred_input": in0, "val_psnr_untrained_model": val0,
               "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               "torch": torch.__version__, "args": vars(args)}
    if device.type == "cuda":
        summary["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    (table_dir / f"train_summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
