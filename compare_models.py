"""
Compare our Qwen3.5-35B-A3B implementation vs HF reference on the same inputs.
Requires torch 2.12.1+cu130 on B300 (SM103) — cu128 grouped_mm crashes there.

Usage:
  python3 compare_models.py --weight-dir /path/to/weights
  python3 compare_models.py --weight-dir /path/to/weights --no-gen
"""
import argparse, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_ours(weight_dir):
    repo_src = Path(__file__).parent / "src"
    sys.path.insert(0, str(repo_src))
    from model import Qwen35MoE, load_weights
    model = Qwen35MoE().to(torch.bfloat16).to("cuda:0").eval()
    load_weights(model, weight_dir, verbose=False)
    return model


def load_hf(weight_dir):
    return AutoModelForCausalLM.from_pretrained(
        weight_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda:0").eval()


def topk_tokens(logits, tok, k=5):
    return [repr(tok.decode([i])) for i in logits.topk(k).indices.tolist()]


def gen_nocache(model, ids, tok, max_new=60, is_ours=True):
    all_ids = ids[0].tolist()
    for _ in range(max_new):
        inp = torch.tensor([all_ids], device="cuda:0")
        with torch.no_grad():
            if is_ours:
                logits, _, _, _ = model(inp)
            else:
                logits = model(inp).logits
        nid = logits[0, -1].argmax().item()
        all_ids.append(nid)
        if nid == tok.eos_token_id:
            break
    return tok.decode(all_ids[ids.shape[1]:], skip_special_tokens=True)


PROMPTS = [
    "The chemical formula of baking soda is",
    "Compute: 13 × 17 =",
    "def fib(n):\n    if n <= 1: return n\n    return",
    "The element with atomic number 79 is",
    "In the year 2157, the dominant programming language is",
    "If the answer is 42, what could the question be?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", default="weights")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--no-gen", action="store_true")
    args = ap.parse_args()

    print(f"torch {torch.__version__} | SM {torch.cuda.get_device_capability(0)}")
    print(f"Weight dir: {args.weight_dir}\n")

    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)

    print("Loading our model...")
    ours = load_ours(args.weight_dir)
    print("  done")

    print("Loading HF model...")
    hf = load_hf(args.weight_dir)
    print("  done\n")

    sep = "=" * 72
    total_cos, top1_match = 0.0, 0

    for prompt in PROMPTS:
        ids = tok.encode(prompt, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            our_logits, _, _, _ = ours(ids)
            hf_logits = hf(ids).logits

        ol = our_logits[0, -1].float()
        hl = hf_logits[0, -1].float()
        cos = F.cosine_similarity(ol.unsqueeze(0), hl.unsqueeze(0)).item()
        our_top5 = topk_tokens(ol, tok)
        hf_top5  = topk_tokens(hl, tok)
        match = our_top5[0] == hf_top5[0]
        total_cos += cos
        top1_match += int(match)

        print(sep)
        print(f"PROMPT : {repr(prompt[:80])}")
        print(f"  cosine={cos:.4f}  top1_match={match}")
        print(f"  ours top-5 : {our_top5}")
        print(f"  HF   top-5 : {hf_top5}")

        if not args.no_gen:
            our_gen = gen_nocache(ours, ids, tok, args.max_new, is_ours=True)
            hf_gen  = gen_nocache(hf,   ids, tok, args.max_new, is_ours=False)
            print(f"  ours gen   : {repr(our_gen[:160])}")
            print(f"  HF   gen   : {repr(hf_gen[:160])}")
        print()

    print(sep)
    n = len(PROMPTS)
    print(f"SUMMARY  avg_cosine={total_cos/n:.4f}  top1_match={top1_match}/{n}")


if __name__ == "__main__":
    main()
