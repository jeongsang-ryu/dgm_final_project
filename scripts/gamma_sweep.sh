#!/usr/bin/env bash
# Run short (200 kimg) fine-tune probes at gamma in {0.5, 2, 8} and report
# the best FID from each. Use to pick the gamma for the full run.
#
# Usage:
#   scripts/gamma_sweep.sh
set -eo pipefail
cd "$(dirname "$0")/.."
source env/activate.sh

data="${DATA:-data/celebvhq256.zip}"
resume="${RESUME:-ckpts/stylegan3-r-ffhqu-256x256.pkl}"
[ -f "$data" ] || { echo "Missing dataset zip: $data"; exit 1; }

for g in 0.5 2 8; do
    outdir="runs/sweep_g${g}"
    if [ -d "$outdir" ]; then
        echo "=== skip sweep g=$g (already exists at $outdir) ==="
        continue
    fi
    mkdir -p "$outdir"
    echo "=== sweep gamma=$g ==="
    (cd stylegan3 && python train.py \
        --outdir "../${outdir}" \
        --cfg stylegan3-r \
        --data "../${data}" \
        --gpus 1 \
        --batch 16 \
        --batch-gpu 4 \
        --gamma "$g" \
        --mirror 1 \
        --aug ada \
        --snap 5 \
        --metrics fid50k_full \
        --kimg 200 \
        --resume "../${resume}")
done

echo "=== summary ==="
for g in 0.5 2 8; do
    j="runs/sweep_g${g}"/*/metric-fid50k_full.jsonl
    if ls $j 2>/dev/null 1>&2; then
        best=$(python -c "
import json,glob
xs=[]
for p in glob.glob('$j'):
    for line in open(p):
        try: xs.append(json.loads(line)['results']['fid50k_full'])
        except: pass
print(min(xs) if xs else 'n/a')
")
        echo "gamma=$g best_fid=$best"
    fi
done
