#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_video> <output_dir> [fps]"
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
FPS="${3:-1}"

mkdir -p "$OUTPUT_DIR"
ffmpeg -i "$INPUT_VIDEO" -vf "fps=$FPS" "$OUTPUT_DIR/frame_%06d.jpg"

