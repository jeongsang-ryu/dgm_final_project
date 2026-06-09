#!/usr/bin/env python
"""Sample N face images from a (pretrained or fine-tuned) LDM and write a flat zip.

Recon use: pretrained CompVis/ldm-celebahq-256, DDIM 200 steps, 1000 images.
Mirrors submit_pack.py's output convention: flat zip, img_0000.png ... in root.

Usage:
  python scripts/ldm_generate.py \
      --model CompVis/ldm-celebahq-256 \
      --count 1000 --steps 200 --batch 20 --seed 0 \
      --name ldm_celebahq_pretrained
"""
import argparse, os, zipfile, tempfile, shutil
import torch
from diffusers import DiffusionPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CompVis/ldm-celebahq-256")
    ap.add_argument("--unet", default=None,
                    help="path to a fine-tuned UNet folder to swap into the base pipeline")
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default="submissions")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ldm_generate] loading {args.model} on {device}")
    pipe = DiffusionPipeline.from_pretrained(args.model)
    if args.unet:
        from diffusers import UNet2DModel
        print(f"[ldm_generate] swapping in fine-tuned UNet from {args.unet}")
        pipe.unet = UNet2DModel.from_pretrained(args.unet)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    workdir = tempfile.mkdtemp(prefix="ldm_gen_")
    n_done = 0
    idx = 0
    while n_done < args.count:
        bs = min(args.batch, args.count - n_done)
        # deterministic per-batch seed for reproducibility.
        # LDM uncond pipeline samples initial latents on CPU, so the generator
        # must be CPU too (else: "Cannot generate a cpu tensor from a cuda generator").
        gen = torch.Generator().manual_seed(args.seed + idx)
        out = pipe(batch_size=bs, num_inference_steps=args.steps,
                   eta=args.eta, generator=gen)
        for img in out.images:
            img.save(os.path.join(workdir, f"img_{n_done:04d}.png"))
            n_done += 1
        idx += 1
        print(f"[ldm_generate] {n_done}/{args.count}")

    os.makedirs(args.outdir, exist_ok=True)
    zpath = os.path.join(args.outdir, f"{args.name}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for i in range(args.count):
            fn = f"img_{i:04d}.png"
            zf.write(os.path.join(workdir, fn), arcname=fn)
    shutil.rmtree(workdir, ignore_errors=True)
    mb = os.path.getsize(zpath) / 1e6
    print(f"[ldm_generate] wrote {zpath} ({mb:.1f} MB, {args.count} imgs)")


if __name__ == "__main__":
    main()
