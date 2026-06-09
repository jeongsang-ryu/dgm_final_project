#!/usr/bin/env python3
"""Generate a side-by-side mosaic of aligned CelebV-HQ frames vs FFHQU samples.

Use this to eyeball whether our alignment crop-style matches what the
pretrained FFHQU-256 checkpoint was trained on — distribution mismatch here
is the #1 source of catastrophic FID in fine-tuning.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stylegan3"))
import dnnlib  # noqa: E402
import legacy  # noqa: E402


def mosaic(imgs_left, imgs_right, out_path, gap=8):
    assert len(imgs_left) == len(imgs_right)
    n = len(imgs_left)
    H, W = imgs_left[0].shape[:2]
    total_w = 2 * W + gap
    total_h = n * H + (n - 1) * gap
    canvas = np.full((total_h, total_w, 3), 255, np.uint8)
    for i, (L, R) in enumerate(zip(imgs_left, imgs_right)):
        y = i * (H + gap)
        canvas[y:y + H, 0:W] = L
        canvas[y:y + H, W + gap:] = R
    PIL.Image.fromarray(canvas).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", default="data/celebvhq_aligned")
    ap.add_argument("--pkl", default="ckpts/stylegan3-r-ffhqu-256x256.pkl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="data/align_sanity.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    aligned_root = Path(args.aligned)
    pngs = list(aligned_root.rglob("*.png"))
    print(f"{len(pngs)} aligned images found")
    if len(pngs) < args.n:
        sys.exit("not enough aligned images")
    sample = random.sample(pngs, args.n)
    lefts = [np.array(PIL.Image.open(p).convert("RGB")) for p in sample]

    device = torch.device("cuda")
    with dnnlib.util.open_url(args.pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    G.eval()
    label = torch.zeros([1, G.c_dim], device=device)
    rights = []
    for s in range(args.n):
        z = torch.from_numpy(np.random.RandomState(10000 + s).randn(1, G.z_dim)).to(device)
        with torch.no_grad():
            img = G(z, label, truncation_psi=1.0, noise_mode="const")
        img = (img.clamp(-1, 1) * 0.5 + 0.5) * 255.0
        rights.append(img.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()[0])

    mosaic(lefts, rights, args.out)
    print(f"wrote {args.out}   left=aligned CelebV-HQ   right=FFHQU samples")


if __name__ == "__main__":
    main()
