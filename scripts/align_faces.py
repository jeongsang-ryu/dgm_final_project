#!/usr/bin/env python3
"""Detect the main face in each frame and crop an FFHQU-style square at 256x256.

FFHQU (used by the pretrained StyleGAN3-R ffhqu-256 ckpt) is "FFHQ Unaligned":
just a square crop around the face bbox, no rotation alignment. This matches
video frames better than the classic FFHQ 5-landmark similarity warp.

Uses insightface SCRFD for detection; picks the largest face; expands to a
square with configurable margin; resizes to 256x256 with Lanczos.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/celebvhq_frames")
    ap.add_argument("--out-dir", default="data/celebvhq_aligned")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.35,
                    help="fraction of bbox side added as padding on each side")
    ap.add_argument("--det-size", type=int, default=320,
                    help="SCRFD detection input size (lower = faster)")
    ap.add_argument("--min-face", type=int, default=128,
                    help="skip faces smaller than this (pixels, longer side)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=-1,
                    help=">=0 uses CUDAExecutionProvider, -1 uses CPU "
                         "(safer: ORT cuBLAS/cuDNN ABI is fragile on this env)")
    args = ap.parse_args()

    from insightface.app import FaceAnalysis

    if args.gpu >= 0:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = args.gpu
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    jpgs = list(in_root.rglob("*.jpg"))
    print(f"{len(jpgs)} frames found")
    if args.limit:
        jpgs = jpgs[: args.limit]

    n_ok = n_skip = n_nofacing = 0
    t0 = time.time()
    for i, frame_path in enumerate(jpgs):
        rel = frame_path.relative_to(in_root)
        out_path = out_root / rel.with_suffix(".png")
        if out_path.exists():
            n_skip += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        faces = app.get(img)
        if not faces:
            n_nofacing += 1
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox
        bw, bh = x2 - x1, y2 - y1
        if max(bw, bh) < args.min_face:
            n_skip += 1
            continue
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        side = max(bw, bh) * (1.0 + 2.0 * args.margin)
        half = side * 0.5
        l, t, r, b = int(round(cx - half)), int(round(cy - half)), \
                     int(round(cx + half)), int(round(cy + half))
        # pad with edge reflection if crop falls outside the frame
        pad_l = max(0, -l); pad_t = max(0, -t)
        pad_r = max(0, r - W); pad_b = max(0, b - H)
        if pad_l or pad_t or pad_r or pad_b:
            img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_REFLECT)
            l += pad_l; r += pad_l; t += pad_t; b += pad_t
        crop = img[t:b, l:r]
        out = cv2.resize(crop, (args.size, args.size),
                         interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(out_path), out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        n_ok += 1
        if n_ok and n_ok % 1000 == 0:
            dt = time.time() - t0
            rate = n_ok / dt
            eta = (len(jpgs) - i) / rate
            print(f"[{n_ok} ok / {i+1} seen] nofacing={n_nofacing} "
                  f"skip={n_skip} rate={rate:.1f}/s eta={eta/60:.1f}min")
            sys.stdout.flush()

    print(f"DONE ok={n_ok} nofacing={n_nofacing} skipped={n_skip} "
          f"elapsed={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
