#!/usr/bin/env bash
# Start the Qwen3.5-35B-A3B inference server.
#
# Usage:
#   ./start.sh                              # weights at ./weights/, port 8000
#   ./start.sh --weight-dir /path/to/weights
#   ./start.sh --port 8080 --device cuda:0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="${WEIGHT_DIR:-$SCRIPT_DIR/weights}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --weight-dir) WEIGHT_DIR="$2"; shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Qwen3.5-35B-A3B Inference Server ==="
echo "  Weights : $WEIGHT_DIR"
echo "  Device  : $DEVICE"
echo "  Port    : $PORT"
echo ""

if [[ ! -f "$WEIGHT_DIR/model.safetensors.index.json" ]]; then
    echo "ERROR: weights not found at $WEIGHT_DIR"
    echo "Run: python3 download_weights.py --dest $WEIGHT_DIR"
    exit 1
fi

pip install -q fastapi 'uvicorn[standard]' safetensors transformers \
    accelerate pydantic 'jinja2>=3.1.0' 2>/dev/null || true

exec python3 "$SCRIPT_DIR/src/server.py" \
    --weight-dir "$WEIGHT_DIR" \
    --device     "$DEVICE" \
    --port       "$PORT"
