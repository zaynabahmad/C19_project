"""Download and unpack the project datasets onto the machine running this script

Sources (official pages):
  GoPro (GOPRO_Large): https://seungjunnah.github.io/Datasets/gopro   (CC BY 4.0, per that page)
  DIV2K:               https://data.vision.ee.ethz.ch/cvl/DIV2K/

Usage (from the repo root):
  python src/fetch_data.py --dataset div2k
  python src/fetch_data.py --dataset gopro
  python src/fetch_data.py --dataset gopro --url <other-mirror-url>   # if the default link fails

"""
import argparse
import os
import sys
import urllib.request
import zipfile

SOURCES = {
    "div2k": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "gopro": "https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip",
}


def download(url: str, dest: str) -> None:
    """Download url to dest, resuming a partial file if one exists."""
    done = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable: file already complete
            print(f"{dest} already complete.")
            return
        raise
    # 206 = server honoured the range request; otherwise start from scratch.
    mode = "ab" if resp.status == 206 else "wb"
    total = resp.headers.get("Content-Length")
    print(f"Downloading {url}\n  -> {dest} (remaining bytes: {total})")
    with open(dest, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    ap.add_argument("--url", default=None, help="override the default download URL")
    ap.add_argument("--out", default="data/raw", help="output directory (default: data/raw)")
    args = ap.parse_args()

    url = args.url or SOURCES[args.dataset]
    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, os.path.basename(url.split("?")[0]) or f"{args.dataset}.zip")

    download(url, zip_path)
    if not zipfile.is_zipfile(zip_path):
        print(f"ERROR: {zip_path} is not a valid zip file (download failed or the link returned an error page).")
        return 1
    print(f"Unpacking {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out)
    print(f"Done. Top-level contents of {args.out}:")
    for name in sorted(os.listdir(args.out)):
        print("  ", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())