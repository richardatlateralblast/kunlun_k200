#!/usr/bin/env bash
# Pipes real frames from an RTSP source into demos/video_pipeline_live.
# No URL or credentials are stored anywhere in this repo -- pass the source
# at run time:
#
#   RTSP_URL='rtsp://user:pass@host:554/path' scripts/run_video_pipeline_live.sh [frames]
#
# A password containing special characters (e.g. '/') must be percent-encoded
# in RTSP_URL (python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' <pass>).
set -euo pipefail

: "${RTSP_URL:?Set RTSP_URL to an rtsp:// source, e.g. RTSP_URL='rtsp://user:pass@host:554/path' $0}"
FRAMES="${1:-6}"

cd "$(dirname "$0")/.."
make demos >/dev/null

ffmpeg -loglevel error -rtsp_transport tcp -i "$RTSP_URL" \
    -vf "fps=1,scale=8:8" -pix_fmt rgb24 -f rawvideo -frames:v "$FRAMES" - \
  | ./demos/video_pipeline_live --frames "$FRAMES"
