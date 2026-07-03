"""
Minimal OpenAI-compatible server for Qwen3.5-35B-A3B.

Implements:
  GET  /health
  POST /v1/chat/completions

Concurrency strategy: each request runs in its own thread with a GPU mutex.
The mutex serialises GPU access (one forward pass at a time) while allowing
multiple requests to overlap CPU work and queue decode steps. This means
N concurrent requests each make progress round-robin rather than waiting
for the entire previous request to finish.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model import Qwen35MoE, load_weights

# ── Request / response schemas ────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "Qwen/Qwen3.5-35B-A3B"
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0

# ── Engine ────────────────────────────────────────────────────────────────────

class Engine:
    """
    Each request gets its own thread. A single GPU mutex serialises forward
    passes so VRAM is never double-booked. Threads yield the mutex between
    every token, so requests at the same concurrency level take turns.
    """
    def __init__(self, model, tokenizer, first_device: str = "cuda:0"):
        self.model       = model
        self.tok         = tokenizer
        self.first_device = first_device
        self.eos_id      = tokenizer.eos_token_id
        self._gpu_lock   = threading.Lock()   # serialises GPU forward passes

    def generate(self, input_ids: torch.Tensor, max_tokens: int,
                 temperature: float, top_p: float) -> list[int]:
        """Run full generation for one request. Blocks caller thread."""
        ids = input_ids.to(self.first_device)
        kvs = states = convs = None
        output_ids: list[int] = []

        # Prefill
        with self._gpu_lock:
            with torch.no_grad():
                logits, kvs, states, convs = self.model(ids, kvs=kvs, states=states, convs=convs)
            next_id = self._sample(logits[0, -1], temperature, top_p)
        output_ids.append(next_id)

        # Decode
        while next_id != self.eos_id and len(output_ids) < max_tokens:
            tok_tensor = torch.tensor([[next_id]], device=self.first_device)
            with self._gpu_lock:
                with torch.no_grad():
                    logits, kvs, states, convs = self.model(
                        tok_tensor, kvs=kvs, states=states, convs=convs)
                next_id = self._sample(logits[0, -1], temperature, top_p)
            output_ids.append(next_id)

        return output_ids

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        logits = logits.float()
        if temperature == 0.0:
            return logits.argmax().item()
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, -1)
            cumprobs = torch.cumsum(probs, -1)
            remove = (cumprobs - probs) > top_p
            sorted_logits[remove] = float('-inf')
            logits = torch.scatter(logits, -1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, -1)
        return torch.multinomial(probs, 1).item()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app  = FastAPI()
_engine: Optional[Engine] = None
_tok:    Optional[AutoTokenizer] = None

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    global _engine, _tok
    if _engine is None:
        raise HTTPException(503, "Model not loaded")

    # Apply chat template (thinking disabled via enable_thinking=False)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        text = _tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        text = _tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    input_ids = _tok.encode(text, return_tensors="pt", add_special_tokens=False)
    prompt_tokens = input_ids.shape[1]

    # Run generation in a thread so the event loop stays unblocked.
    # The engine's GPU mutex serialises forward passes across all threads.
    loop = asyncio.get_event_loop()
    output_ids = await loop.run_in_executor(
        None,
        _engine.generate,
        input_ids,
        req.max_tokens,
        req.temperature,
        req.top_p,
    )

    output_text = _tok.decode(output_ids, skip_special_tokens=True)
    completion_tokens = len(output_ids)
    finish_reason = "stop" if (output_ids and output_ids[-1] == _tok.eos_token_id) else "length"

    return JSONResponse({
        "id":      f"chatcmpl-{uuid.uuid4()}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "Qwen/Qwen3.5-35B-A3B",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": output_text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    })

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _engine, _tok

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", default=str(Path(__file__).parent.parent / "weights"),
                        help="Path to safetensors weights dir")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--host",       default="0.0.0.0")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.weight_dir}...")
    _tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)

    print("Building model...")
    model = Qwen35MoE().to(torch.bfloat16)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print("Loading weights...")
    load_weights(model, args.weight_dir, verbose=False)
    print("Moving to cuda:0...")
    model = model.to("cuda:0").eval()

    print("Starting engine...")
    _engine = Engine(model, _tok, "cuda:0")

    print(f"Server ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
