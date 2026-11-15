"""Select the reproducible dataset subsets and write them to data/subsets/*.json.

Run from the repo root, AFTER the data has been downloaded to data/raw/:
    python -m src.make_subset

GoPro: the official split is by video sequence (train/ and test/ folders).
  - train/val: some WHOLE training sequences are held out for validation, so validation
    frames are never near-duplicates of training frames.
  - every K-th frame (sorted by file name) is kept inside each sequence.
  - test: every K-th frame of every test sequence.
DIV2K: N images sampled with a fixed seed from DIV2K_valid_HR (used only as a Wiener
sanity check).

GoPro has two blurred versions per sharp frame (blur/ and blur_gamma/). The one used is chosen with
--blur-dir (default: blur) and recorded in the output JSON; keep it fixed for ALL experiments.

The script does NOT assume the folder layout: it searches data/raw for a train/ folder whose
sequence sub-folders contain blur/ and sharp/ folders, and stops with a clear message if it
cannot find one.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from src.utils import SEED

IMG_EXT = {".png", ".jpg", ".jpeg"}


def images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def find_gopro_root(raw: Path, blur_dir: str):
    """Return the folder that contains train/ and test/ (sequence folders with blur/ and sharp/)."""
    for train in sorted(raw.rglob("train")):
        if not train.is_dir():
            continue
        seqs = [c for c in train.iterdir() if c.is_dir() and (c / blur_dir).is_dir() and (c / "sharp").is_dir()]
        if seqs and (train.parent / "test").is_dir():
            return train.parent
    return None


def sequences(split_dir: Path, blur_dir: str):
    return sorted(c for c in split_dir.iterdir() if c.is_dir() and (c / blur_dir).is_dir())


def pairs_for_sequence(seq: Path, root: Path, k: int, blur_dir: str):
    out = []
    for i, blur in enumerate(images(seq / blur_dir)):
        if i % k != 0:
            continue
        sharp = seq / "sharp" / blur.name
        if not sharp.exists():
            raise FileNotFoundError(f"No sharp counterpart for {blur}")
        out.append({"blur": str(blur.relative_to(root)), "sharp": str(sharp.relative_to(root))})
    return out


def make_gopro(raw: Path, out: Path, k: int, n_val_seq: int, seed: int, blur_dir: str) -> bool:
    root = find_gopro_root(raw, blur_dir)
    if root is None:
        print(f"ERROR: could not find a GoPro train/ + test/ layout under {raw}.")
        print("Top-level contents:", sorted(p.name for p in raw.iterdir()) if raw.exists() else "(folder missing)")
        return False
    train_seqs, test_seqs = sequences(root / "train", blur_dir), sequences(root / "test", blur_dir)
    val_names = sorted(random.Random(seed).sample([s.name for s in train_seqs], n_val_seq))
    data = {"seed": seed, "k": k, "blur_dir": blur_dir, "root": str(root), "val_sequences": val_names,
            "train": [], "val": [], "test": []}
    for s in train_seqs:
        data["val" if s.name in val_names else "train"] += pairs_for_sequence(s, root, k, blur_dir)
    for s in test_seqs:
        data["test"] += pairs_for_sequence(s, root, k, blur_dir)
    (out / "gopro_subset.json").write_text(json.dumps(data, indent=1))
    print(f"GoPro root: {root} (blur folder: {blur_dir})")
    print(f"  train sequences: {len(train_seqs)} (val held out: {val_names}), test sequences: {len(test_seqs)}")
    print(f"  pairs -> train: {len(data['train'])}, val: {len(data['val'])}, test: {len(data['test'])}")
    return True


def make_div2k(raw: Path, out: Path, n: int, seed: int) -> bool:
    folders = [p for p in raw.rglob("DIV2K_valid_HR") if p.is_dir()]
    if not folders:
        print(f"ERROR: DIV2K_valid_HR not found under {raw}")
        return False
    files = images(folders[0])
    chosen = sorted(random.Random(seed).sample([p.name for p in files], n))
    data = {"seed": seed, "n": n, "root": str(folders[0]), "images": chosen}
    (out / "div2k_subset.json").write_text(json.dumps(data, indent=1))
    print(f"DIV2K: {len(chosen)} of {len(files)} images selected (seed {seed})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/subsets")
    ap.add_argument("--k", type=int, default=5, help="keep every k-th frame (default 5)")
    ap.add_argument("--val-seqs", type=int, default=3, help="training sequences held out for validation")
    ap.add_argument("--div2k-n", type=int, default=25)
    ap.add_argument("--blur-dir", default="blur", choices=["blur", "blur_gamma"], help="which blurred version to use")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = make_gopro(raw, out, a.k, a.val_seqs, a.seed, a.blur_dir)
    ok = make_div2k(raw, out, a.div2k_n, a.seed) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())