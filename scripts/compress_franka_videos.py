#!/usr/bin/env python3
"""
Compress the franka_local videos for faster web loading while keeping clarity.
Requires ffmpeg to be installed and available on PATH.
"""
from pathlib import Path
import shutil
import subprocess
import sys

INPUT_DIR = Path("docs/static/videos/franka_local")
OUTPUT_DIR = INPUT_DIR / "compressed"
VIDEO_EXTS = {".mp4", ".mov"}


def compress_one(src: Path, dst: Path) -> int:
  # Scale down to max 1280px width, keep aspect ratio, reasonable quality/size balance.
  cmd = [
      "ffmpeg",
      "-y",
      "-i",
      str(src),
      "-vf",
      "scale=min(1280\\,iw):-2",
      "-c:v",
      "libx264",
      "-preset",
      "slow",
      "-crf",
      "28",
      "-movflags",
      "+faststart",
      "-c:a",
      "aac",
      "-b:a",
      "96k",
      str(dst),
  ]
  result = subprocess.run(cmd, capture_output=True)
  if result.returncode != 0:
    sys.stderr.write(f"ffmpeg failed for {src.name}:\n{result.stderr.decode(errors='ignore')}\n")
  return result.returncode


def main() -> int:
  if not shutil.which("ffmpeg"):
    sys.stderr.write("ffmpeg is not installed or not on PATH. Please install ffmpeg and try again.\n")
    return 1

  if not INPUT_DIR.exists():
    sys.stderr.write(f"Input directory not found: {INPUT_DIR}\n")
    return 1

  OUTPUT_DIR.mkdir(exist_ok=True)
  videos = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
  if not videos:
    print("No videos found to compress.")
    return 0

  for src in videos:
    dst = OUTPUT_DIR / f"{src.stem}_web.mp4"
    print(f"Compressing {src.name} -> {dst.name} ...")
    code = compress_one(src, dst)
    if code != 0:
      return code

  print(f"Done. Compressed files saved to {OUTPUT_DIR}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
