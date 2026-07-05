"""
Self-contained pure-PyTorch implementation of Qwen3.5-35B-A3B.
Exact numerical match to HF reference implementation.

Architecture:
  40 layers alternating [linear_attn x3, full_attn x1] repeated 10 times.
  linear_attn: Gated Delta Rule (GDR) with L2-normed QK
  full_attn:   GQA with partial RoPE and output gate
  FFN: MoE (256 experts, top-8) + shared expert (sigmoid gate)
"""
import math, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from safetensors import safe_open

# ── Architecture constants (from config.json text_config) ───────────────────
H     = 2048
L     = 40
FAI   = 4                # full_attention_interval
# Full attention
NQ    = 16               # Q heads
NKV   = 2                # KV heads
DH    = 256              # head dim (full)
# Linear attention (GDR)
LKH   = 16               # K/Q heads (before expansion)
LVH   = 32               # V heads (= SSM heads)
LHD   = 128              # head dim (linear, key/value)
CK    = 4                # conv1d kernel
# RoPE
THETA = 10_000_000.0
PROT  = 0.25             # partial_rotary_factor → rot_dim = int(DH*PROT) = 64
# MoE
NE    = 256
TK    = 8
MI    = 512
SI    = 512
EPS   = 1e-6
VOCAB = 248320

def is_full(i): return (i + 1) % FAI == 0

# ── RMS Norm ────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Qwen3.5-MoE RMSNorm: output = rms_norm(x) * (1 + weight).
    Used for input_layernorm, post_attention_layernorm, and final norm.
    Weight stored as offset from 1 (init=0); formula: Qwen3_5MoeRMSNorm in HF."""
    def __init__(self, d, eps=EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(d))  # init 0, effective scale = 1+w

    def forward(self, x):
        dt = x.dtype
        x  = x.float()
        x  = x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * (1.0 + self.weight.float())).to(dt)


class RMSNormGated(nn.Module):
    """Qwen3_5MoeRMSNormGated: rms_norm then weight*x, then silu(gate) multiply.
    Weight init ~1 (standard multiply, no offset). Exact HF formula:
      x = x * rsqrt(mean(x^2) + eps)
      x = weight * x.to(input_dtype)    ← cast before weight mul
      x = x * silu(gate.float())
      return x.to(input_dtype)
    """
    def __init__(self, d, eps=EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, gate):
        dt   = x.dtype
        x    = x.float()
        x    = x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x    = (self.weight.to(dt) * x.to(dt)).float()   # weight multiply at input dtype
        return (x * F.silu(gate.float())).to(dt)

# ── RoPE (partial, for full attention only) ──────────────────────────────────
def build_rope(T, device):
    rot = int(DH * PROT)    # 64
    inv = 1.0 / THETA ** (torch.arange(0, rot, 2, device=device).float() / rot)
    t   = torch.arange(T, device=device).float()
    f   = torch.outer(t, inv)   # [T, 32]
    return f.cos(), f.sin()

def rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)

def apply_rope(q, k, cos, sin):
    """q,k: [B, heads, T, DH]. cos,sin: [T, 32]."""
    rot = cos.shape[-1] * 2
    qr, qp = q[..., :rot], q[..., rot:]
    kr, kp = k[..., :rot], k[..., rot:]
    T = q.shape[2]
    c = torch.cat([cos[:T], cos[:T]], -1).to(q)[None, None]
    s = torch.cat([sin[:T], sin[:T]], -1).to(q)[None, None]
    return (torch.cat([qr*c + rot_half(qr)*s, qp], -1),
            torch.cat([kr*c + rot_half(kr)*s, kp], -1))



# ── Full Attention (GQA + output gate) ──────────────────────────────────────
class FullAttn(nn.Module):
    def __init__(self):
        super().__init__()
        # q_proj outputs 2*NQ*DH: first NQ*DH = query, next NQ*DH = gate
        self.q_proj = nn.Linear(H, 2 * NQ * DH, bias=False)
        self.k_proj = nn.Linear(H, NKV * DH,    bias=False)
        self.v_proj = nn.Linear(H, NKV * DH,    bias=False)
        self.o_proj = nn.Linear(NQ * DH, H,      bias=False)
        self.q_norm = RMSNorm(DH)
        self.k_norm = RMSNorm(DH)

    def forward(self, x, cos, sin, mask=None, kv=None):
        B, T, _ = x.shape
        # Project: q contains [Q, gate] interleaved at head_dim*2 level
        q_full = self.q_proj(x).view(B, T, NQ, DH * 2)
        q, gate = q_full.chunk(2, dim=-1)        # each [B,T,NQ,DH]
        gate = gate.reshape(B, T, -1)            # [B,T,NQ*DH]

        q = self.q_norm(q.transpose(1,2))         # [B,NQ,T,DH] after norm
        k = self.k_norm(self.k_proj(x).view(B,T,NKV,DH).transpose(1,2))
        v = self.v_proj(x).view(B,T,NKV,DH).transpose(1,2)

        q, k = apply_rope(q, k, cos, sin)

        if kv is not None:
            k = torch.cat([kv[0], k], 2)
            v = torch.cat([kv[1], v], 2)
        new_kv = (k, v)

        g = NQ // NKV
        k = k.repeat_interleave(g, 1)
        v = v.repeat_interleave(g, 1)

        scale = DH ** -0.5
        a = (q @ k.transpose(-2,-1)) * scale
        if mask is not None: a = a + mask
        a = F.softmax(a.float(), -1).to(q.dtype)
        o = (a @ v).transpose(1,2).reshape(B, T, NQ*DH)
        # Output gate (attn_output_gate=True)
        o = o * torch.sigmoid(gate.to(o.dtype))
        return self.o_proj(o), new_kv

# ── L2 norm helper ───────────────────────────────────────────────────────────
def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# ── Gated Delta Rule (GDR) linear attention ──────────────────────────────────
class LinearAttn(nn.Module):
    """
    Implements the Gated Delta Rule from the HF transformers source.
    
    g   = -A_log.exp() * softplus(in_proj_a(x) + dt_bias)  [B,T,32]  (negative, log-decay)
    beta = sigmoid(in_proj_b(x))                            [B,T,32]
    Q,K,V from in_proj_qkv after conv+silu
    Q,K are L2-normalized and scaled by 1/sqrt(LHD)
    K is expanded from 16 heads to 32 heads (repeat×2)
    Q is expanded from 16 heads to 32 heads (repeat×2)
    
    Recurrent form (sequential for correctness):
      For each t:
        g_t    (log-decay scalar per head, negative)
        beta_t (interpolation weight per head)
        decay  = exp(g_t)                           scalar
        k_t    = beta_t * k_t  (weighted K)
        state  = decay * state + k_t^T @ (v_t - k_t @ state)   [delta rule]
        y_t    = q_t @ state
    
    This is equivalent to the chunked form with use_qk_l2norm=True.
    """
    QKV = (LKH + LKH + LVH) * LHD   # 16×128 + 16×128 + 32×128 = 8192

    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(H, self.QKV, bias=False)
        self.in_proj_z   = nn.Linear(H, LVH * LHD, bias=False)  # gate → norm
        self.in_proj_a   = nn.Linear(H, LVH, bias=False)         # log-dt
        self.in_proj_b   = nn.Linear(H, LVH, bias=False)         # beta
        self.conv1d      = nn.Conv1d(self.QKV, self.QKV, CK, groups=self.QKV, padding=CK-1, bias=False)
        self.A_log       = nn.Parameter(torch.zeros(LVH))
        self.dt_bias     = nn.Parameter(torch.zeros(LVH))
        self.norm        = RMSNormGated(LHD)
        self.out_proj    = nn.Linear(LVH * LHD, H, bias=False)

    def forward(self, x, state=None, conv_state=None):
        """
        x:          [B, T, H]
        state:      [B, LVH, LHD, LHD]  (GDR state)  or None
        conv_state: [B, QKV, CK-1]                    or None
        """
        B, T, _ = x.shape
        QKV = self.QKV

        z   = self.in_proj_z(x)                            # [B,T,LVH*LHD]
        a   = self.in_proj_a(x)                            # [B,T,LVH=32]
        b   = self.in_proj_b(x)                            # [B,T,LVH=32]
        qkv = self.in_proj_qkv(x)                         # [B,T,8192]

        # Causal depthwise conv1d
        qkv_t = qkv.transpose(1, 2)                        # [B,8192,T]
        if conv_state is not None and T == 1:
            # Single-token decode: use causal_conv1d_update logic.
            # Prepend state, apply conv with no padding, output is the single result.
            combined = torch.cat([conv_state, qkv_t], dim=2)  # [B,8192,CK-1+1=CK]
            new_conv = combined[:, :, -(CK-1):].detach()      # rolling window: drop oldest
            qkv_conv = F.conv1d(
                combined, self.conv1d.weight, self.conv1d.bias,
                padding=0, groups=QKV,
            )                                                  # [B,8192,1]
            qkv_conv = qkv_conv.transpose(1, 2)               # [B,1,8192]
        else:
            # Prefill (or chunked decode with prior state): standard causal conv with padding.
            if conv_state is not None:
                qkv_t = torch.cat([conv_state, qkv_t], dim=2)
            qkv_conv = self.conv1d(qkv_t)                     # [B,8192, pad+T+pad]
            new_conv  = qkv_t[:, :, -(CK-1):].detach()
            offset = conv_state.shape[2] if conv_state is not None else 0
            qkv_conv = qkv_conv[:, :, offset:offset+T].transpose(1, 2)  # [B,T,8192]
        qkv_conv = F.silu(qkv_conv)

        # Split Q, K, V (key_dim=LKH*LHD=2048, value_dim=LVH*LHD=4096)
        q = qkv_conv[:, :, :LKH*LHD]                                    # [B,T,2048]
        k = qkv_conv[:, :, LKH*LHD:2*LKH*LHD]                          # [B,T,2048]
        v = qkv_conv[:, :, 2*LKH*LHD:]                                  # [B,T,4096]

        q = q.view(B, T, LKH, LHD)        # [B,T,16,128]
        k = k.view(B, T, LKH, LHD)        # [B,T,16,128]
        v = v.view(B, T, LVH, LHD)        # [B,T,32,128]

        # Compute g = -A_log.exp() * softplus(a + dt_bias)  [B,T,32]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        # beta = sigmoid(b)  [B,T,32]
        beta = b.sigmoid()

        # Expand Q,K from 16→32 heads (each K head serves 2 V heads)
        q = q.repeat_interleave(2, dim=2)   # [B,T,32,128]
        k = k.repeat_interleave(2, dim=2)   # [B,T,32,128]

        # L2-normalize Q and K (use_qk_l2norm_in_kernel=True)
        scale = LHD ** -0.5
        q = l2norm(q.float()) * scale       # [B,T,32,128] float32
        k = l2norm(k.float())               # [B,T,32,128] float32

        # GDR sequential scan in float32 — matches HF torch_recurrent_gated_delta_rule
        # exactly (it casts q,k,v,beta,g to float32 before the scan and only casts the
        # output back to bf16 at the end). Running this in bf16 causes the recurrent
        # state to accumulate rounding error much faster on repetitive/low-entropy
        # input (state updates barely change step-to-step, so bf16 quantization noise
        # doesn't average out) — verified this compounds into degenerate repetition
        # loops during autoregressive generation, not just a small logit diff.
        #   S = S * g_t                          (decay first)
        #   kv_mem = (S * k_t).sum(dim=-2)       (k_t @ decayed S)
        #   delta = (v_t - kv_mem) * beta_t      (beta on residual)
        #   S = S + k_t[:,:,None] * delta[:,None] (rank-1 update)
        #   y_t = (S * q_t).sum(dim=-2)
        dt = x.dtype
        if state is None:
            S = torch.zeros(B, LVH, LHD, LHD, device=x.device, dtype=torch.float32)
        else:
            S = state.float()

        g    = g.float()
        beta = beta.float()
        v    = v.float()

        ys = []
        for t in range(T):
            g_t    = g[:, t, :].exp().unsqueeze(-1).unsqueeze(-1)  # [B,32,1,1]
            beta_t = beta[:, t, :].unsqueeze(-1)                   # [B,32,1]
            k_t    = k[:, t, :, :]          # [B,32,128]
            v_t    = v[:, t, :, :]          # [B,32,128]
            q_t    = q[:, t, :, :]          # [B,32,128]

            # Decay state first
            S = S * g_t                                             # [B,32,128,128]

            # Memory retrieval from decayed state
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)           # [B,32,128]

            # Delta: (v - k@S) * beta
            delta = (v_t - kv_mem) * beta_t                        # [B,32,128]

            # Rank-1 update
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)        # [B,32,128,128]

            # Output
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)              # [B,32,128]
            ys.append(y_t)

        new_state = S.detach()

        # Cast back to model dtype here (matches HF: torch_recurrent_gated_delta_rule
        # casts core_attn_out to initial_dtype before it ever reaches the gated norm).
        y = torch.stack(ys, dim=1).to(dt)   # [B,T,32,128]
        y = y.reshape(B*T*LVH, LHD)    # flatten for norm: [B*T*32, 128]
        z_flat = z.reshape(B*T*LVH, LHD)
        y = self.norm(y, z_flat)        # RMSNormGated: norm(y) * silu(z), returns x.dtype
        y = y.reshape(B, T, LVH * LHD)

        return self.out_proj(y), new_state, new_conv

# ── MoE FFN ──────────────────────────────────────────────────────────────────
class SharedExpert(nn.Module):
    """Matches HF mlp.shared_expert.{gate_proj,up_proj,down_proj}."""
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, SI, bias=False)
        self.up_proj   = nn.Linear(H, SI, bias=False)
        self.down_proj = nn.Linear(SI, H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Experts(nn.Module):
    """Matches HF mlp.experts.{gate_up_proj,down_proj}."""
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(NE, 2*MI, H))  # [256,1024,2048] #TODO: WHY IS IT UNINITIALIZED DATA  (torch.empty)
        self.down_proj    = nn.Parameter(torch.empty(NE, H,    MI))  # [256,2048, 512]


class MoEFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts         = Experts()
        self.gate            = nn.Linear(H, NE, bias=False)
        self.shared_expert   = SharedExpert()
        self.shared_expert_gate = nn.Linear(H, 1, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        xf = x.reshape(-1, H)   # [N, H]
        N  = xf.shape[0]

        w, idx = torch.topk(self.gate(xf), TK, -1)   # [N, TK]
        w = F.softmax(w, -1).to(x.dtype)              # [N, TK]

        flat_idx = idx.reshape(-1)          # [N*TK]
        flat_w   = w.reshape(-1)            # [N*TK]
        token_rep = xf.unsqueeze(1).expand(N, TK, H).reshape(N * TK, H)

        sort_order     = torch.argsort(flat_idx, stable=True)
        sorted_idx     = flat_idx[sort_order]
        sorted_tokens  = token_rep[sort_order]
        sorted_weights = flat_w[sort_order]

        expert_counts  = torch.bincount(sorted_idx, minlength=NE)
        expert_offsets = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long),
                                    expert_counts.cumsum(0)[:-1]])

        sorted_out = torch.zeros(N * TK, H, device=x.device, dtype=x.dtype)

        for e in range(NE):
            cnt = expert_counts[e].item()
            if cnt == 0:
                continue
            start = expert_offsets[e].item()
            xt = sorted_tokens[start:start+cnt]
            gw, uw = self.experts.gate_up_proj[e].chunk(2, 0)
            h = F.silu(xt @ gw.t()) * (xt @ uw.t())
            h = h @ self.experts.down_proj[e].t()
            sorted_out[start:start+cnt] = sorted_weights[start:start+cnt].unsqueeze(-1) * h

        unsort_order = torch.argsort(sort_order, stable=True)
        out = sorted_out[unsort_order].reshape(N, TK, H).sum(dim=1)

        sg = torch.sigmoid(self.shared_expert_gate(xf))
        out = out + sg * self.shared_expert(xf)
        return out.view(B, T, H)

# ── Decoder Layer ─────────────────────────────────────────────────────────────
class Layer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i    = i
        self.full = is_full(i)
        self.input_layernorm          = RMSNorm(H)
        self.post_attention_layernorm = RMSNorm(H)
        if self.full:
            self.self_attn = FullAttn()
        else:
            self.linear_attn = LinearAttn()
        self.mlp = MoEFFN()

    def forward(self, x, cos=None, sin=None, mask=None,
                kv=None, state=None, conv=None):
        r = x
        h = self.input_layernorm(x)
        if self.full:
            a, new_kv    = self.self_attn(h, cos, sin, mask, kv)
            new_s, new_c = None, None
        else:
            a, new_s, new_c = self.linear_attn(h, state, conv)
            new_kv = None
        x = r + a #residual
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv, new_s, new_c

# ── Full Model ────────────────────────────────────────────────────────────────
class Qwen35MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, H)
        self.layers  = nn.ModuleList([Layer(i) for i in range(L)])
        self.norm    = RMSNorm(H)
        self.lm_head = nn.Linear(H, VOCAB, bias=False)
        self._rope_T = 0
        self.register_buffer('_cos', torch.zeros(1,1), persistent=False)
        self.register_buffer('_sin', torch.zeros(1,1), persistent=False)

    def _ensure_rope(self, T, dev):
        if T > self._rope_T:
            c, s = build_rope(T + 64, dev)
            self._cos, self._sin = c, s
            self._rope_T = T + 64



    @torch.no_grad()
    def forward(self, ids, kvs=None, states=None, convs=None):
        B, T  = ids.shape
        dev   = ids.device
        x     = self.embed_tokens(ids)

        if kvs    is None: kvs    = [None]*L
        if states is None: states = [None]*L
        if convs  is None: convs  = [None]*L

        past  = kvs[3][0].shape[2] if kvs[3] is not None else 0
        Ttot  = T + past
        self._ensure_rope(Ttot, dev)

        mask = torch.full((T, Ttot), float('-inf'), device=dev, dtype=x.dtype)
        for i in range(T): mask[i, :past+i+1] = 0.0
        mask = mask[None, None]

        nkvs, nss, ncs = [], [], []
        for i, layer in enumerate(self.layers):
            x, nk, ns, nc = layer(x, cos=self._cos, sin=self._sin,
                                  mask=mask, kv=kvs[i],
                                  state=states[i], conv=convs[i])
            nkvs.append(nk); nss.append(ns); ncs.append(nc)

        logits = self.lm_head(self.norm(x))
        return logits, nkvs, nss, ncs

# ── Weight Loading ────────────────────────────────────────────────────────────
def _hf_to_our_key(hk: str) -> str | None:
    """Strip HF key prefix to match our model's state_dict keys.

    HF stores weights under 'model.*' (text backbone) or at top-level
    ('lm_head.weight'). Our model mirrors HF naming exactly except we
    have no 'model.' wrapper — everything lives at the top level.
    """
    if hk.startswith('model.visual.') or hk.startswith('mtp.'):
        return None
    # Multi-modal wrapper used by some checkpoints
    if hk.startswith('model.language_model.'):
        hk = hk[len('model.language_model.'):]
    # Strip the 'model.' prefix that wraps the text backbone in HF
    if hk.startswith('model.'):
        hk = hk[len('model.'):]
    return hk


def load_weights(model, weight_dir, verbose=True):
    """Load HF safetensors weights directly — no translation table needed.
    Our attribute names mirror HF exactly; we only strip the 'model.' prefix.
    """
    with open(os.path.join(weight_dir, 'model.safetensors.index.json')) as f:
        wmap = json.load(f)['weight_map']

    shards: dict[str, list[str]] = {}
    for k, v in wmap.items():
        shards.setdefault(v, []).append(k)

    sd = model.state_dict()
    mapped, mismatches = 0, []

    for shard_name in sorted(shards):
        f = safe_open(os.path.join(weight_dir, shard_name), framework='pt', device='cpu')
        for hk in shards[shard_name]:
            mk = _hf_to_our_key(hk)
            if mk is None:
                continue
            if mk not in sd:
                mismatches.append(f'missing: {hk} → {mk}')
                continue
            t = f.get_tensor(hk)
            if t.shape != sd[mk].shape:
                mismatches.append(f'shape mismatch {hk}: hf={t.shape} ours={sd[mk].shape}')
                continue
            sd[mk] = t.to(sd[mk].dtype)
            mapped += 1

    if mismatches:
        print(f'  Issues ({len(mismatches)}):')
        for m in mismatches[:10]:
            print(f'    {m}')
    if verbose:
        print(f'  Mapped {mapped} tensors.')
    model.load_state_dict(sd, strict=False)
    return model


def generate(model, ids, tokenizer, max_new_tokens=50, temperature=1.0,
             top_p=1.0, enable_thinking=False):
    """
    Autoregressive generation with temperature and thinking-mode control.

    Args:
        enable_thinking: If False, prepends a system prompt that suppresses
                         Qwen3's <think> blocks. Qwen3 models default to
                         thinking mode; this must be explicitly disabled.
    """
    if enable_thinking:
        # Thinking mode: model will emit <think>...</think> before answering
        input_ids = ids
    else:
        # Suppress thinking: wrap as chat with system instruction
        # Qwen3 respects /no_think token or system prompt
        msgs = [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {"role": "user",   "content": tokenizer.decode(ids[0].tolist())},
        ]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(text, return_tensors='pt').to(ids.device)

    generated = input_ids.clone()
    kvs = states = convs = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, kvs, states, convs = model(
                generated if kvs is None else generated[:, -1:],
                kvs=kvs, states=states, convs=convs)
        last_logits = logits[0, -1].float()

        if temperature == 0.0:
            next_id = last_logits.argmax(keepdim=True).unsqueeze(0).to(generated.device)
            generated = torch.cat([generated, next_id], -1)
            if next_id.item() == tokenizer.eos_token_id:
                break
            continue

        if temperature != 1.0:
            last_logits = last_logits / temperature

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(last_logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, -1), -1)
            remove = cumprobs - F.softmax(sorted_logits, -1) > top_p
            sorted_logits[remove] = float('-inf')
            last_logits = torch.scatter(last_logits, -1, sorted_idx, sorted_logits)

        probs = F.softmax(last_logits, -1)
        next_id = torch.multinomial(probs, 1).unsqueeze(0).to(generated.device)
        generated = torch.cat([generated, next_id], -1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    new_ids = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)


if __name__ == '__main__':
    import argparse, gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description='Qwen3.5-35B-A3B inference')
    parser.add_argument('--weight-dir',  default='/home/sesterce/qwen35/weights')
    parser.add_argument('--device',      default='cuda:0')
    parser.add_argument('--prompt',      default='The capital of France is')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='0 = greedy, >0 = sampling')
    parser.add_argument('--max-new-tokens', type=int, default=50)
    parser.add_argument('--thinking',    action='store_true',
                        help='Enable Qwen3 thinking mode (off by default)')
    parser.add_argument('--compare-hf',  action='store_true',
                        help='Also load HF model and compare logits')
    args = parser.parse_args()

    WDIR, DEVICE = args.weight_dir, args.device

    print('=== Qwen3.5-35B-A3B Self-Contained PyTorch ===')
    model = Qwen35MoE().to(torch.bfloat16)
    print(f'  Params: {sum(p.numel() for p in model.parameters())/1e9:.3f}B')
    load_weights(model, WDIR, verbose=True)
    model = model.to(DEVICE).eval()

    tok = AutoTokenizer.from_pretrained(WDIR, trust_remote_code=True)
    ids = tok.encode(args.prompt, return_tensors='pt').to(DEVICE)
    print(f'\nPrompt: "{args.prompt}"')

    if args.compare_hf:
        print('\n=== Logit comparison vs HF ===')
        with torch.no_grad():
            my_logits, _, _, _ = model(ids)
        ml = my_logits[0, -1].float().cpu()
        del model; gc.collect(); torch.cuda.empty_cache()

        hf = AutoModelForCausalLM.from_pretrained(
            WDIR, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
        hf.eval()
        with torch.no_grad():
            hl = hf(ids).logits[0, -1].float().cpu()
        del hf; gc.collect(); torch.cuda.empty_cache()


        cos_sim   = F.cosine_similarity(ml.unsqueeze(0), hl.unsqueeze(0)).item()
        max_diff  = (ml - hl).abs().max().item()
        top5_my   = ml.topk(5).indices.tolist()
        top5_hf   = hl.topk(5).indices.tolist()
        print(f'  Cosine similarity: {cos_sim:.6f}')
        print(f'  Max  |logit diff|: {max_diff:.4f}')
        print(f'  My  top-5: {[tok.decode([i]) for i in top5_my]}')
        print(f'  HF  top-5: {[tok.decode([i]) for i in top5_hf]}')
        print(f'  Top-1 match: {top5_my[0] == top5_hf[0]}')
    else:
        print('\nGenerating...')
        out = generate(
            model, ids, tok,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=args.thinking)
        print(f'Output: {out}')
