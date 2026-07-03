# Qwen3.5-35B-A3B — Inference Engine

Self-contained PyTorch reimplementation of [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) with an OpenAI-compatible server and eval suite. Tested on B300 SXM6 (SM10.3, torch 2.12.1+cu130).

## Architecture

- 40 layers: 3× Gated Delta Rule (linear attention) + 1× full GQA, repeated 10×
- GDR: L2-norm QK, bf16 state, sequential scan
- GQA: 16Q / 2KV heads, head\_dim 256, output gate
- MoE FFN: 256 experts, top-8 routed + 1 shared expert, hidden 2048
- 34.66B total params, ~3B active per token, bfloat16
- Requires torch 2.12.1+cu130 on B300 (cu128 `grouped_mm` crashes on SM10.3)

## Verified results

- Weight mapping: 693/693 (100%)
- Logit cosine vs HF reference: **avg 0.9824**
- Top-1 token match: **5/6** prompts

## Usage

```bash
bash setup.sh                                    # create .venv, install deps, download weights (~67 GB)
bash setup.sh --skip-weights --weight-dir /path  # skip download if weights already present
bash start.sh --weight-dir ./weights             # start server on :8000
.venv/bin/python -m eval.check_server            # smoke test
.venv/bin/python compare_models.py --weight-dir ./weights   # verify vs HF
.venv/bin/python -m eval.correctness.run_correctness        # GSM8K-CoT (200 problems)
.venv/bin/python -m eval.throughput.run_throughput          # throughput benchmark
```

## Files

```
src/model.py       — model implementation (GDR + GQA + MoE)
src/server.py      — OpenAI-compatible chat completions server
compare_models.py  — logit + generation comparison vs HF reference
download_weights.py
setup.sh / start.sh
eval/check_server.py
eval/correctness/run_correctness.py
eval/throughput/run_throughput.py
tests/test_model.py
```
