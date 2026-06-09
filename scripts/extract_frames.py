#!/usr/bin/env python3
"""Stream mp4 clips out of data/celebvhq_raw/videos.tar and extract 1 fps JPGs.

Resumable: clips whose first output frame already exists are skipped.
"""
import argparse
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def extract_clip(mp4_bytes: bytes, out_prefix: str, fps: float,
                 tmp_path: str = "/tmp/_extract_clip.mp4") -> int:
    """Run ffmpeg on an mp4, return number of frames written.

    mp4 files from CelebV-HQ put the moov atom at the end, so ffmpeg can't
    read them from stdin — it needs a seekable file. We write to a fixed
    tmp path (overwritten each call) to avoid per-clip filesystem churn.
    """
    with open(tmp_path, "wb") as f:
        f.write(mp4_bytes)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        f"{out_prefix}_%04d.jpg",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return -1
    base = Path(out_prefix).name
    parent = Path(out_prefix).parent
    return sum(1 for p in parent.glob(f"{base}_*.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default="data/celebvhq_raw/videos.tar")
    ap.add_argument("--out", default="data/celebvhq_frames")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="stop after N clips")
    ap.add_argument("--max-frames-per-clip", type=int, default=4,
                    help="keep at most this many frames per clip (picks evenly spaced)")
    ap.add_argument("--log", default="data/celebvhq_frames/_extract.log")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(args.log, "a", buffering=1)

    tar_path = Path(args.tar)
    if not tar_path.exists():
        sys.exit(f"tar not found: {tar_path}")

    n_clips = n_skipped = n_ok = n_fail = total_frames = 0
    t0 = time.time()

    with tarfile.open(tar_path, "r|") as tf:  # streaming mode
        for member in tf:
            if not member.isfile() or not member.name.endswith(".mp4"):
                continue
            n_clips += 1
            clip_id = Path(member.name).stem
            # shard by first 2 chars of clip id to avoid 35k-file dir
            shard = out_root / clip_id[:2]
            shard.mkdir(exist_ok=True)
            out_prefix = str(shard / clip_id)
            marker = shard / f"{clip_id}.done"
            if marker.exists():
                n_skipped += 1
                continue
            if args.limit and n_clips > args.limit:
                break
            f = tf.extractfile(member)
            if f is None:
                n_fail += 1
                continue
            data = f.read()
            n_frames = extract_clip(data, out_prefix, args.fps)
            if n_frames <= 0:
                n_fail += 1
                logf.write(f"FAIL {clip_id}\n")
                continue

            if args.max_frames_per_clip and n_frames > args.max_frames_per_clip:
                frames = sorted(shard.glob(f"{clip_id}_*.jpg"))
                keep_idx = {
                    round(i * (n_frames - 1) / (args.max_frames_per_clip - 1))
                    for i in range(args.max_frames_per_clip)
                }
                kept = 0
                for i, fr in enumerate(frames):
                    if i in keep_idx:
                        kept += 1
                    else:
                        fr.unlink()
                n_frames = kept

            marker.touch()
            total_frames += n_frames
            n_ok += 1
            if n_ok % 200 == 0:
                dt = time.time() - t0
                print(f"[{n_ok} ok / {n_clips} seen] frames={total_frames} "
                      f"fail={n_fail} skip={n_skipped} elapsed={dt:.0f}s")
                sys.stdout.flush()

    dt = time.time() - t0
    print(f"DONE clips_seen={n_clips} ok={n_ok} fail={n_fail} skipped={n_skipped} "
          f"frames={total_frames} elapsed={dt:.0f}s")
    logf.close()


if __name__ == "__main__":
    main()
