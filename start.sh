#!/usr/bin/env bash
# Start the Qwen3.5-35B-A3B inference server.
# Uses ~/venv (created by setup.sh) with accelerate for multi-GPU dispatch.
# Usage: ./start.sh [--weight-dir PATH] [--port PORT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="${WEIGHT_DIR:-$SCRIPT_DIR/weights}"
PORT="${PORT:-8000}"
VENV="${VENV:-$SCRIPT_DIR/.venv}"
PYTHON="$VENV/bin/python"

while [[ $# -gt 0 ]]; do
    case $1 in
        --weight-dir) WEIGHT_DIR="$2"; shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: venv not found at $VENV — run: bash setup.sh --skip-weights"
    exit 1
fi

if [[ ! -f "$WEIGHT_DIR/model.safetensors.index.json" ]]; then
    echo "ERROR: weights not found at $WEIGHT_DIR — run: bash setup.sh"
    exit 1
fi

N_GPUS=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "=== Qwen3.5-35B-A3B Inference Server ==="
echo "  Weights : $WEIGHT_DIR"
echo "  Port    : $PORT"
echo "  GPUs    : $N_GPUS"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/src/server.py" \
    --weight-dir "$WEIGHT_DIR" \
    --port       "$PORT"
