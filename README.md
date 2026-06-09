# DGM Spring 2026 — CelebV-HQ Face Generation

Fine-tuning **StyleGAN3-R** (primary) and a **latent diffusion model** (LDM,
secondary) to generate 256×256 faces matching the **CelebV-HQ** distribution,
scored by the rank-average of FID / IS / KID / TopP&R.

This repo contains the **inference + training + evaluation code** only.
Datasets, model weights, and generated image zips are **not** tracked (see
`.gitignore`); reproduce them with the steps below.

---

## 1. Environment

Single GPU (developed on an RTX 4080 Super, 16 GB). Two environments:

**(a) conda — StyleGAN3 track** (PyTorch 2.1.2 / CUDA 12.1, plus the build tools
StyleGAN3 needs to JIT-compile its CUDA ops):

```bash
conda env create -f env/environment.yml      # creates env "dgm"
source env/activate.sh                        # activates dgm + sets CUDA_HOME/CC/CXX
```

**(b) venv — LDM + evaluation track** (inherits the conda env's PyTorch, adds
`diffusers`/`accelerate`/`top-pr`/`pytorch-fid`):

```bash
python -m venv --system-site-packages env/ldm_venv
source env/ldm_venv/bin/activate
pip install -r env/requirements-ldm.txt
```

---

## 2. Dataset (CelebV-HQ → 256×256 aligned frames)

```bash
# 2.1 download source videos (HF mirror) -> data/celebvhq_raw/videos.tar
bash scripts/download_mirror.sh

# 2.2 decode frames at 1 fps, max 4 per clip -> data/celebvhq_frames/
python scripts/extract_frames.py

# 2.3 detect + box-crop + resize faces to 256 (InsightFace) -> data/celebvhq_aligned/
python scripts/align_faces.py
python scripts/align_sanity.py        # optional: overlay QA vs FFHQU crop

# 2.4 pack into the StyleGAN3 dataset zip -> data/celebvhq256.zip
#     (uses stylegan3/dataset_tool.py; see script header for the exact command)
```

Result: **132,551** aligned frames from ≈13,400 clips. No external faces are
used (academic-integrity compliant).

---

## 3. Training

**StyleGAN3-R** — fine-tune from the public FFHQU-256 checkpoint
(`ckpts/stylegan3-r-ffhqu-256x256.pkl`). `--cbase=16384` is required to match
that checkpoint.

```bash
# scripts/train.sh <tag> <gamma> <kimg>
bash scripts/train.sh main 2 1500            # primary run -> runs/main_g2/

# continuation probe at lower R1 (resume from a snapshot via RESUME=)
RESUME=runs/main_g2/00000-*/network-snapshot-001400.pkl \
  bash scripts/train.sh ftg1 1 2200
```

**LDM** — fine-tune the UNet of `CompVis/ldm-celebahq-256` (VQ-VAE frozen,
ε-prediction, latent scale 0.18215, EMA):

```bash
source env/ldm_venv/bin/activate
python scripts/ldm_finetune.py --outdir runs/ldm_ft --batch 16 --lr 1e-5 \
    --ema-decay 0.9995 --max-hours 8 --save-every 2000 --sample-every 1000
```

---

## 4. Inference — generate a submission (reproducible from seeds)

**StyleGAN3-R** (1,000 images, flat zip, seeds `0..999`):

```bash
python scripts/submit_pack.py \
    --pkl runs/main_g2/00000-*/network-snapshot-001400.pkl \
    --psi 1.0 --seed-start 0 --count 1000 --noise-mode const \
    --name my_submission
# -> submissions/my_submission.zip
```

**LDM** (1,000 images, 200 DDIM steps, seed 0):

```bash
source env/ldm_venv/bin/activate
python scripts/ldm_generate.py --unet runs/ldm_ft/unet-final-ema \
    --count 1000 --steps 200 --seed 0 --name my_ldm_submission
```

> **Final-submission seeds (mandatory).** The StyleGAN images are generated with
> `--seed-start 0 --count 1000 --noise-mode const`, i.e. seeds **0–999**; the LDM
> images use `--seed 0` with 200 DDIM steps.
