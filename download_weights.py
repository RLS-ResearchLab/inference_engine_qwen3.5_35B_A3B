"""
Download Qwen/Qwen3.5-35B-A3B weights from HuggingFace (~67 GB).

Usage:
    python download_weights.py --dest weights/
    python download_weights.py --dest weights/ --token hf_xxx
"""
import argparse, os
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--dest",  default="weights", help="Local directory to save weights")
p.add_argument("--token", default=None,      help="HuggingFace token (optional, increases rate limits)")
a = p.parse_args()

os.makedirs(a.dest, exist_ok=True)
print(f"Downloading Qwen/Qwen3.5-35B-A3B → {a.dest}")
path = snapshot_download(
    repo_id="Qwen/Qwen3.5-35B-A3B",
    local_dir=a.dest,
    ignore_patterns=["*.bin", "original/*"],
    token=a.token,
)
print(f"Done: {path}")
