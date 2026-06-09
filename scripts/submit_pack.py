#!/usr/bin/env python3
"""Generate exactly 1000 images from a StyleGAN3 pkl and pack them into a zip.

Output layout (flat, required by the leaderboard):
    submission.zip
        img_0000.png
        img_0001.png
        ...
        img_0999.png

Usage:
    python scripts/submit_pack.py --pkl ckpts/stylegan3-r-ffhqu-256x256.pkl \
        --psi 0.85 --seed-start 0 --name ffhqu_psi085
"""
import argparse
import csv
import hashlib
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import PIL.Image
import torch

ROOT = Path(__file__).resolve().parent.parent
SG3 = ROOT / "stylegan3"
sys.path.insert(0, str(SG3))

import dnnlib                     # noqa: E402
import legacy                     # noqa: E402


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--psi", type=float, required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--name", required=True, help="submission name (no .zip)")
    ap.add_argument("--outdir", default=str(ROOT / "submissions"))
    ap.add_argument("--log", default=str(ROOT / "submissions" / "log.csv"))
    ap.add_argument("--noise-mode", default="const", choices=["const", "random", "none"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{args.name}.zip"
    if zip_path.exists():
        sys.exit(f"output already exists: {zip_path}")

    print(f"Loading {args.pkl} ...")
    device = torch.device(args.device)
    with dnnlib.util.open_url(args.pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()

    t0 = time.time()
    label = torch.zeros([1, G.c_dim], device=device)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for i in range(args.count):
            seed = args.seed_start + i
            z = torch.from_numpy(
                np.random.RandomState(seed).randn(1, G.z_dim)
            ).to(device)
            with torch.no_grad():
                img = G(z, label, truncation_psi=args.psi, noise_mode=args.noise_mode)
            img = (img.clamp(-1, 1) * 0.5 + 0.5) * 255.0
            img = img.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()[0]
            buf = io.BytesIO()
            PIL.Image.fromarray(img, "RGB").save(buf, format="PNG",
                                                 optimize=False, compress_level=3)
            zf.writestr(f"img_{i:04d}.png", buf.getvalue())
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  {i+1}/{args.count}  {rate:.1f} img/s  "
                      f"zip={zip_path.stat().st_size/1024/1024:.1f} MB")

    dt = time.time() - t0
    size_mb = zip_path.stat().st_size / 1024 / 1024
    pkl_hash = sha256_of(Path(args.pkl))
    print(f"Wrote {zip_path}  {size_mb:.1f} MB  in {dt:.1f}s")

    log_path = Path(args.log)
    new_file = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "name", "zip", "size_mb", "pkl", "pkl_sha16",
                        "psi", "seed_start", "count", "noise_mode",
                        "rank_fid", "rank_is", "rank_kid", "rank_toppr",
                        "fid", "is", "kid", "toppr", "notes"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                    args.name, str(zip_path), f"{size_mb:.1f}",
                    args.pkl, pkl_hash, args.psi, args.seed_start, args.count,
                    args.noise_mode, "", "", "", "", "", "", "", "", ""])
    print(f"Logged to {log_path}")


if __name__ == "__main__":
    main()
