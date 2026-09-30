"""Descriptive diagnostic: does slot 0 act as an attention sink?

For each initial state, run the bootstrap forward with a replica of
`QwenRuntime.step` that exposes attention weights, and report per-layer mean
(over query heads) attention mass on slot 0 plus attention entropy. The replica
is self-checked against `runtime.step` logits on a cloned state.
"""
import argparse, json, math
from pathlib import Path

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from kv_dreaming.runtime import QwenRuntime
from kv_dreaming.seeding import initialize
from kv_dreaming.fragments import initialize_fragment

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--model", required=True); ap.add_argument("--revision", required=True)
ap.add_argument("--calibration", default="/runs/calib/qwen3-4b-base-cuda-v2.pt")
ap.add_argument("--manifest", default="data/fragments/sources.json")
A = ap.parse_args()
torch.use_deterministic_algorithms(True)
R = QwenRuntime.load(model_id=A.model, revision=A.revision, device="cuda", local_files_only=True)
cal = torch.load(A.calibration, map_location="cpu", weights_only=True)


@torch.inference_mode()
def step_with_attention(token_id, state):
    """Mirror of runtime.step (same ops/order) that also returns attention weights."""
    p = state.next_position
    h = R.model.model.embed_tokens(torch.tensor([[token_id]], device=R.device))
    cos, sin = R.model.model.rotary_emb(h, torch.tensor([[p]], device=R.device))
    bias = torch.tensor(state.counts + [1], device=R.device, dtype=torch.float32).log()
    mass0, ent = [], []
    for i, layer in enumerate(R.model.model.layers):
        res = h; x = layer.input_layernorm(h); a = layer.self_attn
        q = a.q_norm(a.q_proj(x).view(1, 1, -1, R.head_dim)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(1, 1, -1, R.head_dim)).transpose(1, 2)
        v = a.v_proj(x).view(1, 1, -1, R.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        state.keys[i] = torch.cat((state.keys[i], k), 2); state.values[i] = torch.cat((state.values[i], v), 2)
        kk = repeat_kv(state.keys[i], a.num_key_value_groups); vv = repeat_kv(state.values[i], a.num_key_value_groups)
        scores = torch.matmul(q, kk.transpose(2, 3)) * a.scaling  # same op order as runtime.step
        w = torch.softmax(scores.float() + bias[None, None, None, :], dim=-1).to(q.dtype)
        wf = w.float()[0, :, 0, :]  # heads x keys
        mass0.append(float(wf[:, 0].mean())); ent.append(float((-(wf.clamp_min(1e-30)) * wf.clamp_min(1e-30).log()).sum(-1).mean()))
        out = torch.matmul(w, vv).transpose(1, 2).reshape(1, 1, -1).contiguous()
        h = res + a.o_proj(out); h = h + layer.mlp(layer.post_attention_layernorm(h))
    state.positions.append(float(p)); state.counts.append(1); state.next_position += 1
    logits = R.model.lm_head(R.model.model.norm(h))[0, -1].float()
    return logits, mass0, ent


rows = []
SINK_EQ = []
conds = [("soft", {}), ("random", {"alpha": 0.25}), ("random", {"alpha": 1.0}),
         ("fragment", {"beta": 0.0, "sink": "none"}), ("fragment", {"beta": 0.0, "sink": "bos"}),
         ("fragment", {"beta": 0.25, "sink": "none"}), ("fragment", {"beta": 0.25, "sink": "bos"})]
for seed in (11, 23, 47, 89):
    for kind, kw in conds:
        if kind == "fragment":
            st, _ = initialize_fragment(R, A.manifest, seed=seed, beta=kw["beta"], calibration=cal, sink=kw["sink"])
            if kw["sink"] == "bos":  # slot 0 must equal a fresh BOS-only prefill, bitwise
                bos = R.prefill(token_ids=[R.model.config.bos_token_id])
                SINK_EQ.append(all(torch.equal(st.keys[L][:, :, :1], bos.keys[L]) and torch.equal(st.values[L][:, :, :1], bos.values[L])
                                   for L in range(len(bos.keys))))
        else:
            a = kw.get("alpha", 1.0)
            st = initialize(R, kind, seed=seed, prefix_length=64, calibration=cal, alpha_k=a, alpha_v=a)
        ref = R.step(R.model.config.bos_token_id, st.clone())
        logits, m0, en = step_with_attention(R.model.config.bos_token_id, st)
        rows.append({"seed": seed, "kind": kind, **kw, "replica_max_logit_diff": float((logits - ref).abs().max()),
                     "slot0_mass_mean": sum(m0) / len(m0), "slot0_mass_by_layer": m0,
                     "attn_entropy_mean": sum(en) / len(en), "prefix_len": len(st.positions) - 1})
Path(A.out).parent.mkdir(parents=True, exist_ok=True)
Path(A.out).write_text(json.dumps(rows, indent=1))
agg = {}
for r in rows:
    key = r["kind"] + "".join(f" {k}={r[k]}" for k in ("alpha", "beta", "sink") if k in r)
    agg.setdefault(key, []).append(r)
print("sink slot0 bitwise == BOS prefill:", sum(SINK_EQ), "/", len(SINK_EQ))
print("max replica logit diff:", max(r["replica_max_logit_diff"] for r in rows))
for key, rs in agg.items():
    m = [r["slot0_mass_mean"] for r in rs]; e = [r["attn_entropy_mean"] for r in rs]
    late = [sum(r["slot0_mass_by_layer"][4:]) / len(r["slot0_mass_by_layer"][4:]) for r in rs]
    print(f"{key:36s} slot0 mass mean {sum(m)/len(m):.3f} (range {min(m):.3f}-{max(m):.3f}); layers>=4 {sum(late)/len(late):.3f}; attn entropy {sum(e)/len(e):.2f}")
