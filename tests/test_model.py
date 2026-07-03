"""
Test suite for Qwen3.5-35B-A3B self-contained PyTorch implementation.
Compares our model against the HuggingFace reference on multiple axes.

Usage:
    python tests/test_model.py --weight-dir /home/sesterce/qwen35/weights
"""

import argparse
import sys
import time
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import Qwen35MoE, load_weights, L, H, VOCAB, NQ, NKV, DH, LVH, LHD, generate

WDIR   = "/home/sesterce/qwen35/weights"
DEVICE = "cuda:0"


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_our_model(wdir, device):
    m = Qwen35MoE().to(torch.bfloat16)
    load_weights(m, wdir, verbose=False)
    return m.to(device).eval()


def load_hf_model(wdir, device):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        wdir, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    return hf.eval()


def load_tokenizer(wdir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(wdir, trust_remote_code=True)


# ── Individual tests ──────────────────────────────────────────────────────────

def test_shape_sanity():
    """Model outputs correct tensor shapes."""
    print("\n[1] Shape sanity (CPU, no weights)")
    m = Qwen35MoE()
    ids = torch.randint(0, VOCAB, (1, 7))
    with torch.no_grad():
        logits, kvs, states, convs = m(ids)

    assert logits.shape == (1, 7, VOCAB), f"logits shape {logits.shape}"
    assert states[0].shape == (1, LVH, LHD, LHD), f"state[0] {states[0].shape}"
    assert convs[0].shape == (1, (16+16+32)*LHD, 3), f"conv[0] {convs[0].shape}"
    assert kvs[3] is not None and kvs[3][0].shape == (1, NKV, 7, DH)
    # Linear attn layers have no kv
    assert kvs[0] is None
    print("  PASS — logits, kv-cache, ssm-state, conv-state all correct shapes")
    return True


def test_weight_loading(wdir):
    """All 693 HF safetensors tensors map correctly (no shape mismatches)."""
    print("\n[2] Weight loading integrity")
    import json, re
    from safetensors import safe_open

    with open(Path(wdir) / "model.safetensors.index.json") as f:
        wmap = json.load(f)["weight_map"]

    m = Qwen35MoE().to(torch.bfloat16)
    sd_before = {k: v.clone() for k, v in m.state_dict().items()}
    load_weights(m, wdir, verbose=False)
    sd_after = m.state_dict()

    changed = sum(1 for k in sd_after if not torch.equal(sd_before[k], sd_after[k]))
    print(f"  {changed} tensors changed from random init after loading")
    assert changed >= 690, f"Expected >=690 changed tensors, got {changed}"
    print(f"  PASS — {changed}/693 tensors loaded successfully")
    return True


def test_norm_formula(wdir):
    """Verifies the (1+weight) RMSNorm formula matches HF."""
    print("\n[3] RMSNorm (1+weight) formula")
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        wdir, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    hf.eval()
    my = Qwen35MoE().to(torch.bfloat16)
    load_weights(my, wdir, verbose=False)
    my.eval()

    ids = torch.tensor([[760, 6511, 314, 9338, 369]])
    with torch.no_grad():
        x_hf = hf.model.embed_tokens(ids)
        x_my = my.embed_tokens(ids)
        n_hf = hf.model.layers[0].input_layernorm(x_hf).float()
        n_my = my.layers[0].input_layernorm(x_my).float()

    diff = (n_hf - n_my).abs().max().item()
    assert diff == 0.0, f"LayerNorm diff = {diff} (expected 0.0)"
    print(f"  PASS — LayerNorm diff = {diff} (exact match)")
    del hf, my
    gc.collect()
    return True


def test_gdr_components(wdir):
    """GDR input projections (g, beta, conv) AND scan output match HF layer-by-layer."""
    print("\n[4] GDR linear attention — projections + scan output vs HF")
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        wdir, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    hf.eval()
    my = Qwen35MoE().to(torch.bfloat16)
    load_weights(my, wdir, verbose=False)
    my.eval()

    ids = torch.tensor([[760, 6511, 314, 9338, 369]])
    with torch.no_grad():
        x = my.layers[0].input_layernorm(my.embed_tokens(ids))
        hf_la = hf.model.layers[0].linear_attn
        my_la = my.layers[0].linear_attn

        # ── Input projections (basic sanity) ──────────────────────────────────
        qkv_hf = F.silu(hf_la.conv1d(hf_la.in_proj_qkv(x).transpose(1,2))[:,:,:x.shape[1]]).transpose(1,2)
        qkv_my = F.silu(my_la.conv1d(my_la.in_proj_qkv(x).transpose(1,2))[:,:,:x.shape[1]]).transpose(1,2)
        conv_diff = (qkv_hf.float() - qkv_my.float()).abs().max().item()

        a_hf = hf_la.in_proj_a(x); b_hf = hf_la.in_proj_b(x)
        a_my = my_la.in_proj_a(x); b_my = my_la.in_proj_b(x)
        g_hf  = -hf_la.A_log.float().exp() * F.softplus(a_hf.float() + hf_la.dt_bias.float())
        g_my  = -my_la.A_log.float().exp() * F.softplus(a_my.float() + my_la.dt_bias.float())
        g_diff    = (g_hf - g_my).abs().max().item()
        beta_diff = (b_hf.sigmoid() - b_my.sigmoid()).abs().max().item()

        assert conv_diff == 0.0, f"conv diff = {conv_diff}"
        assert g_diff    == 0.0, f"g diff    = {g_diff}"
        assert beta_diff == 0.0, f"beta diff = {beta_diff}"

        # ── Full linear_attn block output vs HF ──────────────────────────────
        # Our block output
        our_out, _, _ = my_la(x)

        # HF block output (no recurrent state = same as prefill)
        hf_out = hf_la(x, use_cache=False, output_attentions=False)[0]

        scan_diff = (hf_out.float() - our_out.float()).abs().max().item()
        scan_cos  = F.cosine_similarity(
            hf_out.reshape(1,-1).float(), our_out.reshape(1,-1).float()).item()

    assert scan_diff < 1e-2, f"GDR scan output max diff = {scan_diff:.6f} (expected <0.01)"
    assert scan_cos > 0.999, f"GDR scan cosine = {scan_cos:.6f} (expected >0.999)"

    print(f"  PASS — conv={conv_diff}, g={g_diff}, beta={beta_diff}")
    print(f"  PASS — GDR scan output: max_diff={scan_diff:.6f}, cosine={scan_cos:.6f}")
    del hf, my
    gc.collect()
    return {"scan_max_diff": scan_diff, "scan_cosine": scan_cos}


def test_logit_comparison(wdir, device):
    """Full forward pass logits match HF (top-5 identical)."""
    print("\n[5] Full forward pass — logit comparison")
    from transformers import AutoTokenizer

    tok = load_tokenizer(wdir)
    prompt = "The capital of France is"
    ids = tok.encode(prompt, return_tensors="pt").to(device)

    # Our model
    model = load_our_model(wdir, device)
    with torch.no_grad():
        our_logits, _, _, _ = model(ids)
    ml = our_logits[0, -1].float().cpu()
    del model; gc.collect(); torch.cuda.empty_cache()

    # HF reference
    hf = load_hf_model(wdir, device)
    with torch.no_grad():
        hl = hf(ids).logits[0, -1].float().cpu()
    del hf; gc.collect(); torch.cuda.empty_cache()

    cos  = F.cosine_similarity(ml.unsqueeze(0), hl.unsqueeze(0)).item()
    mxd  = (ml - hl).abs().max().item()
    t5_m = ml.topk(5).indices.tolist()
    t5_h = hl.topk(5).indices.tolist()

    assert t5_m == t5_h, f"top-5 mismatch: ours={t5_m} hf={t5_h}"
    assert cos > 0.95, f"cosine similarity {cos:.4f} < 0.95"
    print(f"  PASS — cosine={cos:.4f}, max_diff={mxd:.4f}, top-5={t5_m}")
    return {"cosine": cos, "max_diff": mxd, "top5_ours": t5_m, "top5_hf": t5_h}


def test_multi_prompt(wdir, device):
    """Top-1 prediction matches HF on 6 diverse prompts."""
    print("\n[6] Multi-prompt top-1 match")
    from transformers import AutoTokenizer

    prompts = [
        "The capital of France is",
        "What is 2 + 2?",
        "Once upon a time",
        "The quick brown fox",
        "In machine learning, a transformer is",
        "The speed of light is approximately",
    ]
    tok = load_tokenizer(wdir)

    model = load_our_model(wdir, device)
    our_preds = []
    for p in prompts:
        ids = tok.encode(p, return_tensors="pt").to(device)
        with torch.no_grad():
            logits, _, _, _ = model(ids)
        our_preds.append(logits[0, -1].argmax().item())
    del model; gc.collect(); torch.cuda.empty_cache()

    hf = load_hf_model(wdir, device)
    hf_preds = []
    for p in prompts:
        ids = tok.encode(p, return_tensors="pt").to(device)
        with torch.no_grad():
            hf_preds.append(hf(ids).logits[0, -1].argmax().item())
    del hf; gc.collect(); torch.cuda.empty_cache()

    results = []
    all_match = True
    for p, ours, hfr in zip(prompts, our_preds, hf_preds):
        match = ours == hfr
        if not match: all_match = False
        ot = tok.decode([ours]); ht = tok.decode([hfr])
        status = "✓" if match else "✗"
        print(f"    {status} \"{p}\" → ours={repr(ot)} hf={repr(ht)}")
        results.append({"prompt": p, "ours": ot, "hf": ht, "match": match})

    assert all_match, "Not all prompts matched"
    print(f"  PASS — {len(prompts)}/{len(prompts)} prompts matched")
    return results


def test_kv_cache_consistency(wdir, device):
    """
    Two checks:
    (a) Our prefill logits match our incremental logits (internal consistency).
    (b) Our incremental last-token logits match HF's (cross-model correctness).
    This prevents reward hacking where both modes share the same bug.
    """
    print("\n[7] KV-cache consistency — internal + vs HF")
    tok = load_tokenizer(wdir)
    prompt = "The capital of France is"
    ids = tok.encode(prompt, return_tensors="pt").to(device)

    model = load_our_model(wdir, device)
    with torch.no_grad():
        # Full prefill
        logits_full, _, _, _ = model(ids)
        ml_full = logits_full[0, -1].float().cpu()

        # Incremental token-by-token
        kvs = states = convs = None
        for t in range(ids.shape[1]):
            logits_inc, kvs, states, convs = model(
                ids[:, t:t+1], kvs=kvs, states=states, convs=convs)
        ml_inc = logits_inc[0, -1].float().cpu()
    del model; gc.collect(); torch.cuda.empty_cache()

    # (a) Internal consistency
    top1_full = ml_full.argmax().item()
    top1_inc  = ml_inc.argmax().item()
    cos_int   = F.cosine_similarity(ml_full.unsqueeze(0), ml_inc.unsqueeze(0)).item()
    assert top1_full == top1_inc, (
        f"Internal: prefill top-1={tok.decode([top1_full])} != "
        f"incremental top-1={tok.decode([top1_inc])}")
    assert cos_int > 0.999, f"Internal cosine = {cos_int:.6f} (expected >0.999)"
    print(f"  Internal — top-1 match ✓, cosine={cos_int:.6f}")

    # (b) Incremental vs HF reference
    hf = load_hf_model(wdir, device)
    with torch.no_grad():
        hl = hf(ids).logits[0, -1].float().cpu()
    del hf; gc.collect(); torch.cuda.empty_cache()

    top1_hf  = hl.argmax().item()
    cos_hf   = F.cosine_similarity(ml_inc.unsqueeze(0), hl.unsqueeze(0)).item()
    t5_inc   = ml_inc.topk(5).indices.tolist()
    t5_hf    = hl.topk(5).indices.tolist()

    assert top1_inc == top1_hf, (
        f"vs HF: incremental top-1={tok.decode([top1_inc])} != "
        f"hf top-1={tok.decode([top1_hf])}")
    assert cos_hf > 0.95, f"Incremental vs HF cosine = {cos_hf:.6f} (expected >0.95)"
    assert t5_inc == t5_hf, f"vs HF: top-5 mismatch: ours={t5_inc} hf={t5_hf}"

    print(f"  vs HF  — top-1 match ✓, cosine={cos_hf:.6f}, top-5={t5_inc}")
    print("  PASS")
    return {"cos_internal": cos_int, "cos_vs_hf": cos_hf}


def test_generate_no_thinking(wdir, device):
    """Generation with enable_thinking=False produces output without <think> blocks."""
    print("\n[8] Generation with thinking disabled")
    from transformers import AutoTokenizer

    tok = load_tokenizer(wdir)
    model = load_our_model(wdir, device)

    ids = tok.encode("What is 2 + 2?", return_tensors="pt").to(device)
    output = generate(
        model, ids, tok,
        max_new_tokens=30,
        temperature=0.0,
        enable_thinking=False)

    del model; gc.collect(); torch.cuda.empty_cache()

    assert "<think>" not in output, f"Output contains <think> block: {output}"
    assert len(output.strip()) > 0, "Empty output"
    print(f"  Output: {repr(output[:80])}")
    print("  PASS — no <think> block in output")
    return {"output": output}


# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", default=WDIR)
    parser.add_argument("--device",     default=DEVICE)
    parser.add_argument("--skip-gpu",   action="store_true", help="Skip GPU tests")
    args = parser.parse_args()

    results = {}
    failures = []

    def run(name, fn, *a, **kw):
        try:
            t0 = time.time()
            out = fn(*a, **kw)
            results[name] = {"status": "PASS", "time_s": round(time.time()-t0, 1), "detail": out}
        except Exception as e:
            import traceback
            results[name] = {"status": "FAIL", "error": str(e)}
            failures.append(name)
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print("=" * 60)
    print("Qwen3.5-35B-A3B PyTorch Implementation — Test Suite")
    print("=" * 60)

    run("shape_sanity",      test_shape_sanity)
    run("weight_loading",    test_weight_loading,    args.weight_dir)
    run("norm_formula",      test_norm_formula,      args.weight_dir)
    run("gdr_components",    test_gdr_components,    args.weight_dir)

    if not args.skip_gpu:
        run("logit_comparison",  test_logit_comparison,     args.weight_dir, args.device)
        run("multi_prompt",      test_multi_prompt,          args.weight_dir, args.device)
        run("kv_cache",          test_kv_cache_consistency,  args.weight_dir, args.device)
        run("generate_no_think", test_generate_no_thinking,  args.weight_dir, args.device)

    print("\n" + "=" * 60)
    print(f"Results: {len(results)-len(failures)}/{len(results)} passed")
    for name, r in results.items():
        t = f"({r['time_s']}s)" if "time_s" in r else ""
        print(f"  {'✓' if r['status']=='PASS' else '✗'} {name} {t}")
    if failures:
        print(f"\nFailed: {failures}")
        sys.exit(1)

    # Save JSON results
    out_path = Path(__file__).parent.parent / "results" / "test_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
