#!/usr/bin/env bash
# Pipes real frames from an RTSP source into demos/cifar10_live, which
# classifies each frame with a small CNN trained on the public CIFAR-10
# dataset (see scripts/train_cifar.py). No URL or credentials are stored
# anywhere in this repo -- pass the source at run time:
#
#   RTSP_URL='rtsp://user:pass@host:554/path' scripts/run_cifar10_live.sh [frames]
#
# A password containing special characters (e.g. '/') must be percent-encoded
# in RTSP_URL (python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' <pass>).
#
# Requires demos/cifar10_weights/ (the trained model's weight files) to
# already exist -- see scripts/train_cifar.py.
set -euo pipefail

: "${RTSP_URL:?Set RTSP_URL to an rtsp:// source, e.g. RTSP_URL='rtsp://user:pass@host:554/path' $0}"
FRAMES="${1:-5}"

cd "$(dirname "$0")/.."
test -d demos/cifar10_weights || { echo "missing demos/cifar10_weights/ -- train the model first (scripts/train_cifar.py)" >&2; exit 1; }
make demos/cifar10_live >/dev/null

ffmpeg -loglevel error -rtsp_transport tcp -i "$RTSP_URL" \
    -vf "fps=1,scale=32:32" -pix_fmt rgb24 -f rawvideo -frames:v "$FRAMES" - \
  | ./demos/cifar10_live --frames "$FRAMES" --weights-dir demos/cifar10_weights
