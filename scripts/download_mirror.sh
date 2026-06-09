#!/usr/bin/env bash
# Downloads the SwayStar123/CelebV-HQ videos.tar (~41.86 GB) and celebvhq_info.json
# to data/celebvhq_raw/. Resumes on interruption.
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/celebvhq_raw
cd data/celebvhq_raw

# info JSON is tiny — via HF (small, fine)
source /home/js/dgm_final/env/activate.sh
python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download('SwayStar123/CelebV-HQ', 'celebvhq_info.json',
                    repo_type='dataset', local_dir='.', local_dir_use_symlinks=False)
print('info json:', p)
PY

# Big file via wget -c for resume; HF returns 302 to CDN, which wget follows.
echo "Downloading videos.tar (resume supported)..."
wget -c --show-progress --progress=bar:force:noscroll \
     -O videos.tar \
     "https://huggingface.co/datasets/SwayStar123/CelebV-HQ/resolve/main/videos.tar"

echo "Done. Sizes:"
ls -lh videos.tar celebvhq_info.json
