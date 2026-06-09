#!/usr/bin/env python
"""Fine-tune the UNet of CompVis/ldm-celebahq-256 on CelebV-HQ aligned frames.

Latent convention (verified against the inference pipeline):
    z0 = vqvae.encode(x).latents * scaling_factor   (scaling_factor = 0.18215)
    inference does  z_vq = z_diff / scaling_factor ; image = vqvae.decode(z_vq)
Objective: epsilon-prediction DDPM MSE (DDIM/DDPM share the forward process).

VQ-VAE is frozen; only the UNet (+EMA) trains. Designed for a single 16 GB GPU:
bf16 autocast + gradient checkpointing. Saves UNet checkpoints + EMA + QA sample
grids periodically, and stops on a wall-clock cap so there is time left to
generate/evaluate afterwards.

Usage:
  python scripts/ldm_finetune.py --outdir runs/ldm_ft --batch 16 \
      --lr 1e-5 --max-hours 8 --save-every 2000 --sample-every 1000
"""
import argparse, os, time, glob, random, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import DiffusionPipeline, DDPMScheduler
from diffusers.training_utils import EMAModel

SCALE = 0.18215
BASE = "CompVis/ldm-celebahq-256"


def list_images(root, cache):
    if os.path.isfile(cache):
        with open(cache) as f:
            paths = [l.strip() for l in f if l.strip()]
        if paths:
            return paths
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(dp, fn))
    paths.sort()
    with open(cache, "w") as f:
        f.write("\n".join(paths))
    return paths


class FaceLatentDS(Dataset):
    def __init__(self, paths, mirror=True):
        self.paths = paths
        self.mirror = mirror

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BICUBIC)
        a = np.asarray(img, dtype=np.float32)
        if self.mirror and random.random() < 0.5:
            a = a[:, ::-1, :].copy()
        t = torch.from_numpy(a).permute(2, 0, 1) / 127.5 - 1.0
        return t


@torch.no_grad()
def sample_grid(pipe, ema, unet, n, steps, device, path):
    """Quick QA grid from EMA weights. Best-effort; never raises into training."""
    try:
        ema.store(unet.parameters())
        ema.copy_to(unet.parameters())
        unet.eval()
        imgs = pipe(batch_size=n, num_inference_steps=steps,
                    generator=torch.Generator().manual_seed(0)).images
        cols = int(n ** 0.5)
        rows = (n + cols - 1) // cols
        W = H = 256
        grid = Image.new("RGB", (cols * W, rows * H))
        for k, im in enumerate(imgs):
            grid.paste(im, ((k % cols) * W, (k // cols) * H))
        grid.save(path)
    except Exception as e:
        print(f"[sample] skipped: {e}")
    finally:
        ema.restore(unet.parameters())
        unet.train()


def save_ckpt(unet, ema, outdir, tag):
    d = os.path.join(outdir, f"unet-{tag}")
    unet.save_pretrained(d)
    # EMA weights as a separate folder
    ema.store(unet.parameters())
    ema.copy_to(unet.parameters())
    unet.save_pretrained(os.path.join(outdir, f"unet-{tag}-ema"))
    ema.restore(unet.parameters())
    print(f"[ckpt] saved {d} (+ema)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/celebvhq_aligned")
    ap.add_argument("--outdir", default="runs/ldm_ft")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)
    device = "cuda"

    paths = list_images(args.data, os.path.join(args.outdir, "filelist.txt"))
    print(f"[data] {len(paths)} images")
    ds = FaceLatentDS(paths)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)

    pipe = DiffusionPipeline.from_pretrained(BASE)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    vae = pipe.vqvae.eval().requires_grad_(False)
    unet = pipe.unet
    try:
        unet.enable_gradient_checkpointing()
        print("[unet] gradient checkpointing enabled")
    except Exception as e:
        print(f"[unet] no gradient checkpointing ({e}); relying on small batch")
    unet.train()

    sched_cfg = dict(pipe.scheduler.config)
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    T = noise_sched.config.num_train_timesteps
    pred_type = noise_sched.config.prediction_type
    print(f"[sched] T={T} prediction_type={pred_type}")
    assert pred_type == "epsilon", f"unexpected prediction_type {pred_type}"

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=0.0)
    ema = EMAModel(unet.parameters(), decay=args.ema_decay)

    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({**vars(args), "base": BASE, "scale": SCALE,
                   "T": T, "pred_type": pred_type}, f, indent=2)

    logf = open(os.path.join(args.outdir, "train.log"), "a")
    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()

    log(f"[start] batch={args.batch} ga={args.grad_accum} lr={args.lr} "
        f"max_hours={args.max_hours} ema={args.ema_decay}")
    t0 = time.time()
    step = 0
    loss_ema = None
    opt.zero_grad(set_to_none=True)
    stop = False
    while not stop:
        for x in dl:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                z0 = vae.encode(x).latents * SCALE
            noise = torch.randn_like(z0)
            t = torch.randint(0, T, (z0.shape[0],), device=device).long()
            zt = noise_sched.add_noise(z0, noise, t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = unet(zt, t).sample
            loss = F.mse_loss(pred.float(), noise.float()) / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.step(unet.parameters())

            l = loss.item() * args.grad_accum
            loss_ema = l if loss_ema is None else 0.98 * loss_ema + 0.02 * l
            step += 1

            if step % 50 == 0:
                el = time.time() - t0
                ips = step * args.batch / el
                vram = torch.cuda.max_memory_allocated() / 1e9
                log(f"step {step:6d}  loss {loss_ema:.4f}  "
                    f"{ips:.1f} img/s  {el/3600:.2f}h  peakVRAM {vram:.1f}GB")
            if step % args.sample_every == 0:
                # small grid to stay well under VRAM during training
                sample_grid(pipe, ema, unet, 4, 50, device,
                            os.path.join(args.outdir, "samples", f"step_{step:06d}.png"))
            if step % args.save_every == 0:
                # rolling checkpoint (overwritten) for crash safety + disk thrift
                save_ckpt(unet, ema, args.outdir, "latest")
                with open(os.path.join(args.outdir, "latest_step.txt"), "w") as f:
                    f.write(str(step))
            if step % 20000 == 0:
                # permanent milestone for later comparison
                save_ckpt(unet, ema, args.outdir, f"m{step//1000:03d}k")
            if step >= args.max_steps or (time.time() - t0) > args.max_hours * 3600:
                stop = True
                break

    save_ckpt(unet, ema, args.outdir, "final")
    log(f"[done] steps={step} elapsed={ (time.time()-t0)/3600:.2f}h")
    logf.close()


if __name__ == "__main__":
    main()
