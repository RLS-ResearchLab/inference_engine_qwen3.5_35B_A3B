#!/usr/bin/env bash
# One-shot setup for Qwen3.5-35B-A3B inference engine.
# Run once on a fresh sesterce/H100 machine.
#
# Usage:
#   bash setup.sh                        # installs deps + downloads weights
#   bash setup.sh --skip-weights         # installs deps only
#   bash setup.sh --weight-dir /path     # custom weight destination (default: ./weights)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHT_DIR="$SCRIPT_DIR/weights"
SKIP_WEIGHTS=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-weights) SKIP_WEIGHTS=1; shift ;;
        --weight-dir)   WEIGHT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo " Qwen3.5-35B-A3B Inference Engine — Setup"
echo "================================================================"
echo "  Script dir : $SCRIPT_DIR"
echo "  Weight dir : $WEIGHT_DIR"
echo ""

# ── 1. Python deps ────────────────────────────────────────────────────────────
echo "[1/3] Installing Python dependencies..."

# Detect CUDA version to pick the right PyTorch build
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "12.4")
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

# Map to PyTorch wheel index
if   [[ "$CUDA_MAJOR" -ge 13 ]]; then WHL_IDX="cu128"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 6 ]]; then WHL_IDX="cu126"
elif [[ "$CUDA_MAJOR" -eq 12 ]]; then WHL_IDX="cu124"
else WHL_IDX="cu121"
fi

echo "  Detected CUDA $CUDA_VER → using torch wheel index: $WHL_IDX"

# Install torch first (skip if already correct CUDA version)
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
if [[ -z "$TORCH_CUDA" ]]; then
    echo "  Installing PyTorch..."
    pip install -q torch --index-url "https://download.pytorch.org/whl/${WHL_IDX}"
else
    echo "  PyTorch already installed (CUDA $TORCH_CUDA)"
fi

# Install remaining deps
pip install -q \
    safetensors \
    "transformers>=4.45" \
    accelerate \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    "jinja2>=3.1.0" \
    huggingface_hub \
    "lm-eval[api]>=0.4.4" \
    aiohttp \
    tabulate \
    tqdm \
    numpy

echo "  Done."
echo ""

# ── 2. Verify CUDA is accessible ─────────────────────────────────────────────
echo "[2/3] Verifying CUDA..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
n = torch.cuda.device_count()
print(f'  {n} GPU(s) available:')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'    [{i}] {p.name}  {p.total_memory//1024**3} GB')
"
echo ""

# ── 3. Download weights ───────────────────────────────────────────────────────
if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
    echo "[3/3] Skipping weight download (--skip-weights)"
else
    echo "[3/3] Downloading Qwen/Qwen3.5-35B-A3B weights (~67 GB)..."
    echo "  Destination: $WEIGHT_DIR"
    python3 "$SCRIPT_DIR/download_weights.py" --dest "$WEIGHT_DIR"
fi

echo ""
echo "================================================================"
echo " Setup complete."
echo ""
echo " Next steps:"
echo "   Start server : ./start.sh --weight-dir $WEIGHT_DIR"
echo "   Check server : python3 -m eval.check_server --base-url http://localhost:8000"
echo "   Correctness  : python3 -m eval.correctness.run_correctness --base-url http://localhost:8000"
echo "   Throughput   : python3 -m eval.throughput.run_throughput --base-url http://localhost:8000"
echo "================================================================"
