# Qwen3.5-35B-A3B Inference Engine

Self-contained pure-PyTorch inference engine for [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) with an OpenAI-compatible HTTP API.

## Quick Start (fresh server)

```bash
git clone git@github.com:RLS-ResearchLab/inference_engine_qwen3.5_35B_A3B.git
cd inference_engine_qwen3.5_35B_A3B

# Install deps + download weights (~67 GB from HuggingFace)
bash setup.sh

# Start server on port 8000
./start.sh
```

## Files

```
src/
  model.py          Pure-PyTorch Qwen3.5-35B-A3B (no custom CUDA)
  server.py         OpenAI-compatible FastAPI server
eval/
  check_server.py             API conformance check
  correctness/run_correctness.py   GSM8K-CoT eval (gate: >=87.5%)
  throughput/run_throughput.py     Throughput benchmark
tests/
  test_model.py     Correctness tests vs HF reference
setup.sh            One-shot: install deps + download weights
start.sh            Launch inference server
download_weights.py Download weights from HuggingFace
requirements.txt    Python dependencies
```

## Server API

```
GET  /health                   → 200 {"status": "ok"}
POST /v1/chat/completions      → OpenAI-compatible response
```

### Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-35B-A3B",
       "messages": [{"role": "user", "content": "What is 2+2?"}],
       "max_tokens": 64, "temperature": 0.0}'
```

## Evaluation

```bash
# Conformance
python3 -m eval.check_server --base-url http://localhost:8000

# GSM8K-CoT correctness (gate: >=87.5% exact match)
python3 -m eval.correctness.run_correctness --base-url http://localhost:8000

# Throughput benchmark
python3 -m eval.throughput.run_throughput --base-url http://localhost:8000
```

## Architecture

| Property | Value |
|---|---|
| Layers | 40 (3× linear + 1× full attention, repeated 10×) |
| Linear attn | Gated Delta Rule (GDR) with L2-norm QK |
| Full attn | GQA (16Q/2KV heads, head_dim=256) + output gate |
| FFN | MoE (256 experts, top-8 routed) + 1 shared expert |
| Hidden dim | 2048 |
| Total params | 34.66B |
| Active params | ~3B per token |
| Dtype | bfloat16 |

## Verified correctness

- Cosine similarity vs HF reference: **0.9586**
- Top-5 logits: **identical** to HF
- Weight loading: **693/693** tensors mapped
