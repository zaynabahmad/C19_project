"""Classical baseline: Wiener-filter deconvolution.

Two stages (run from the repo root, after src/make_subset.py has written data/subsets/*.json):

  python -m src.wiener_baseline --stage div2k    # SANITY CHECK: synthetic blur, kernel KNOWN
  python -m src.wiener_baseline --stage gopro    # BASELINE: real blur, kernel UNKNOWN

Wiener deconvolution (frequency domain, per colour channel):
    X_hat = conj(H) / (|H|^2 + K) * Y
H = Fourier transform of the blur kernel (PSF), Y = blurred image, K = noise-to-signal constant.

GoPro blur is real, varies across the image and has no ground-truth kernel. So the GoPro baseline
uses ONE assumed linear-motion kernel (length, angle) plus K for the whole dataset, chosen on the
VALIDATION pairs by grid search (maximum mean PSNR). The test pairs are only used once, at the end.
This is a deliberately simple non-blind baseline; it is expected to be weak on GoPro.

Outputs: results/tables/*.csv, results/tables/wiener_summary.json, results/figures/wiener_gopro_examples.png
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import fft as sfft

from src.metrics import psnr, ssim
from src.utils import SEED, set_seed

TABLES = Path("results/tables")
FIGURES = Path("results/figures")


# ----------------------------------------------------------------------------- helpers
def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Normalised linear motion-blur kernel (a line segment), odd-sized, bilinear splatting."""
    size = int(np.ceil(length)) | 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    th = np.deg2rad(angle_deg)
    dx, dy = np.cos(th), -np.sin(th)
    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, max(int(length * 8), 8)):
        x, y = c + t * dx, c + t * dy
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for xx, yy, w in ((x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)),
                          (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy)):
            if 0 <= xx < size and 0 <= yy < size:
                k[yy, xx] += w
    return k / k.sum()


def make_filter(psf: np.ndarray, img_shape, K: float):
    """Return (G, pad). The image is reflect-padded by `pad` pixels before filtering and cropped
    afterwards: the FFT assumes a circular image, and without padding the wrap-around at the borders
    creates strong artefacts (checked on test images: padding gave 1-3 dB better PSNR)."""
    pad = int(2 * max(psf.shape))
    pad = min(pad, img_shape[0] - 1, img_shape[1] - 1)
    return wiener_filter(psf, (img_shape[0] + 2 * pad, img_shape[1] + 2 * pad), K), pad


def wiener_filter(psf: np.ndarray, shape, K: float) -> np.ndarray:
    """Frequency response G = conj(H) / (|H|^2 + K) for images of the given (H, W) shape."""
    big = np.zeros(shape, np.float32)
    kh, kw = psf.shape
    big[:kh, :kw] = psf
    big = np.roll(big, (-(kh // 2), -(kw // 2)), axis=(0, 1))  # kernel centre at the origin
    H = sfft.rfft2(big)
    return (np.conj(H) / (np.abs(H) ** 2 + K)).astype(np.complex64)


def deconvolve(img: np.ndarray, G: np.ndarray, pad: int) -> np.ndarray:
    h, w = img.shape[:2]
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ph, pw = p.shape[:2]
    out = np.stack([sfft.irfft2(sfft.rfft2(p[..., c]) * G, s=(ph, pw)) for c in range(3)], axis=-1)
    out = out[pad:pad + h, pad:pad + w]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def update_summary(key: str, value: dict):
    path = TABLES / "wiener_summary.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    TABLES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


# ----------------------------------------------------------------------------- DIV2K sanity check
def run_div2k(args):
    cfg = json.load(open("data/subsets/div2k_subset.json"))
    root = Path(cfg["root"])
    if not root.exists():
        raise SystemExit(f"{root} not found. Download the data and re-run src.make_subset first.")
    names = cfg["images"][: args.limit] if args.limit else cfg["images"]
    rng = np.random.default_rng(args.seed)
    Ks = [1e-3, 1e-2, 3e-2, 1e-1]
    rows = []
    t0 = time.time()
    for name in names:
        sharp = load_rgb(root / name)
        angle = float(rng.uniform(0, 180))
        psf = motion_kernel(args.div2k_length, angle)
        blurred = cv2.filter2D(sharp, -1, psf, borderType=cv2.BORDER_REFLECT)
        blurred = np.clip(blurred + rng.normal(0, args.noise, blurred.shape).astype(np.float32), 0, 1)
        row = {"image": name, "angle_deg": round(angle, 1),
               "psnr_blurred": psnr(sharp, blurred), "ssim_blurred": ssim(sharp, blurred)}
        for K in Ks:  # the TRUE kernel is given to the filter in this sanity check
            G, pad = make_filter(psf, sharp.shape[:2], K)
            rec = deconvolve(blurred, G, pad)
            row[f"psnr_K{K:g}"], row[f"ssim_K{K:g}"] = psnr(sharp, rec), ssim(sharp, rec)
        rows.append(row)
        print(f"  {name}: blurred {row['psnr_blurred']:.2f} dB -> " +
              ", ".join(f"K={K:g}: {row[f'psnr_K{K:g}']:.2f}" for K in Ks))
    write_csv(TABLES / "wiener_div2k.csv", rows)
    summary = {"n_images": len(rows), "kernel_length": args.div2k_length, "noise_sigma": args.noise,
               "seed": args.seed, "limit": args.limit,
               "mean_psnr_blurred": mean(rows, "psnr_blurred"), "mean_ssim_blurred": mean(rows, "ssim_blurred")}
    for K in Ks:
        summary[f"mean_psnr_K{K:g}"], summary[f"mean_ssim_K{K:g}"] = mean(rows, f"psnr_K{K:g}"), mean(rows, f"ssim_K{K:g}")
    update_summary("div2k_sanity_check", summary)
    print(json.dumps(summary, indent=1), f"\n({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------- GoPro baseline
def run_gopro(args):
    cfg = json.load(open("data/subsets/gopro_subset.json"))
    root = Path(cfg["root"])
    if not root.exists():
        raise SystemExit(f"{root} not found. Download the data and re-run src.make_subset first.")
    val, test = cfg["val"], cfg["test"]
    if args.limit:
        test = test[: args.limit]
    rng = np.random.default_rng(args.seed)
    tune_pairs = [val[i] for i in sorted(rng.choice(len(val), size=min(args.tune_n, len(val)), replace=False))]
    tune_imgs = [(load_rgb(root / p["blur"]), load_rgb(root / p["sharp"])) for p in tune_pairs]
    shape = tune_imgs[0][0].shape[:2]

    grid = list(itertools.product(args.lengths, args.angles, args.Ks))
    print(f"Tuning on {len(tune_imgs)} validation pairs, {len(grid)} configs (length, angle, K) ...")
    t0, tuning = time.time(), []
    for L, ang, K in grid:
        G, pad = make_filter(motion_kernel(L, ang), shape, K)
        score = float(np.mean([psnr(s, deconvolve(b, G, pad)) for b, s in tune_imgs]))
        tuning.append({"length": L, "angle_deg": ang, "K": K, "val_mean_psnr": score})
    write_csv(TABLES / "wiener_gopro_tuning.csv", tuning)
    best = max(tuning, key=lambda r: r["val_mean_psnr"])
    base_val = float(np.mean([psnr(s, b) for b, s in tune_imgs]))
    print(f"Best on validation: {best}  (blurred input: {base_val:.2f} dB)  [{time.time() - t0:.0f}s]")

    best_psf = motion_kernel(best["length"], best["angle_deg"])
    G, pad = make_filter(best_psf, shape, best["K"])
    rows, examples = [], []
    print(f"Evaluating on {len(test)} TEST pairs ...")
    for i, p in enumerate(test):
        blur, sharp = load_rgb(root / p["blur"]), load_rgb(root / p["sharp"])
        if blur.shape[:2] != shape:
            G_i, pad_i = make_filter(best_psf, blur.shape[:2], best["K"])
        else:
            G_i, pad_i = G, pad
        rec = deconvolve(blur, G_i, pad_i)
        rows.append({"blur": p["blur"], "psnr_blurred": psnr(sharp, blur), "ssim_blurred": ssim(sharp, blur),
                     "psnr_wiener": psnr(sharp, rec), "ssim_wiener": ssim(sharp, rec)})
        if i < 3:
            examples.append(np.concatenate([blur, rec, sharp], axis=1))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(test)}")
    write_csv(TABLES / "wiener_gopro_test.csv", rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    strip = (np.concatenate(examples, axis=0) * 255).round().astype(np.uint8)
    cv2.imwrite(str(FIGURES / "wiener_gopro_examples.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    summary = {"n_test": len(rows), "blur_dir": cfg.get("blur_dir"), "seed": args.seed, "limit": args.limit,
               "tuned_on_n_val_pairs": len(tune_imgs), "best_config": best,
               "mean_psnr_blurred": mean(rows, "psnr_blurred"), "mean_ssim_blurred": mean(rows, "ssim_blurred"),
               "mean_psnr_wiener": mean(rows, "psnr_wiener"), "mean_ssim_wiener": mean(rows, "ssim_wiener")}
    update_summary("gopro_test", summary)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["div2k", "gopro"], required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N images (QUICK TEST; do not report)")
    ap.add_argument("--tune-n", type=int, default=20, help="validation pairs used for the grid search")
    ap.add_argument("--lengths", type=float, nargs="+", default=[3, 7, 11, 15, 21, 31])
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 45, 90, 135])
    ap.add_argument("--Ks", type=float, nargs="+", default=[1e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--div2k-length", type=float, default=15, help="true kernel length for the sanity check")
    ap.add_argument("--noise", type=float, default=0.01, help="noise std (images in [0,1]) for the sanity check")
    args = ap.parse_args()
    set_seed(args.seed)
    run_div2k(args) if args.stage == "div2k" else run_gopro(args)


if __name__ == "__main__":
    main()
