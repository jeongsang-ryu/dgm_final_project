#!/usr/bin/env python3
"""One-off: rebuild FFHQU baseline zips with PNG compress_level=9.

submit_pack.py uses compress_level=3 for speed (each zip in ~12s). For the
baseline we re-run with level 9 to shave ~30 MB per zip — in case the
leaderboard upload is timing out on the larger baseline zips.
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import PIL.Image
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stylegan3"))
import dnnlib  # noqa: E402
import legacy  # noqa: E402

PKL = ROOT / "ckpts" / "stylegan3-r-ffhqu-256x256.pkl"
OUT_DIR = ROOT / "submissions"
PSIS = [0.7, 0.85, 1.0]


def main():
    device = torch.device("cuda")
    print(f"Loading {PKL} ...")
    with dnnlib.util.open_url(str(PKL)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device).eval()
    label = torch.zeros([1, G.c_dim], device=device)

    for psi in PSIS:
        tag = f"{psi:g}".replace(".", "")
        name = f"ffhqu_baseline_v2_psi{tag}"
        zip_path = OUT_DIR / f"{name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        t0 = time.time()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for i in range(1000):
                z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
                with torch.no_grad():
                    img = G(z, label, truncation_psi=psi, noise_mode="const")
                img = (img.clamp(-1, 1) * 0.5 + 0.5) * 255.0
                img = img.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()[0]
                buf = io.BytesIO()
                PIL.Image.fromarray(img, "RGB").save(buf, "PNG",
                                                     optimize=True, compress_level=9)
                zf.writestr(f"img_{i:04d}.png", buf.getvalue())
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"  {name}: {size_mb:.1f} MB  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
