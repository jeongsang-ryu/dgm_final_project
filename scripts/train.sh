#!/usr/bin/env bash
# Launch StyleGAN3-R fine-tune on CelebV-HQ 256 dataset zip.
#
# Usage:
#   scripts/train.sh <run-tag> [gamma] [kimg]
# Example:
#   scripts/train.sh main 2 2000
set -eo pipefail
cd "$(dirname "$0")/.."

tag="${1:-main}"
gamma="${2:-2}"
kimg="${3:-2000}"
data="${DATA:-data/celebvhq256.zip}"
resume="${RESUME:-ckpts/stylegan3-r-ffhqu-256x256.pkl}"

source env/activate.sh

[ -f "$data" ] || { echo "Missing dataset zip: $data"; exit 1; }
[ -f "$resume" ] || { echo "Missing resume ckpt: $resume"; exit 1; }

outdir="runs/${tag}_g${gamma}"
mkdir -p "$outdir"
echo "=== tag=$tag gamma=$gamma kimg=$kimg data=$data resume=$resume ==="
echo "=== outdir=$outdir ==="

cd stylegan3
exec python train.py \
    --outdir "../${outdir}" \
    --cfg stylegan3-r \
    --data "../${data}" \
    --gpus 1 \
    --batch 16 \
    --batch-gpu 4 \
    --gamma "$gamma" \
    --mirror 1 \
    --aug ada \
    --snap 10 \
    --metrics fid50k_full \
    --kimg "$kimg" \
    --cbase 16384 \
    --resume "../${resume}"
