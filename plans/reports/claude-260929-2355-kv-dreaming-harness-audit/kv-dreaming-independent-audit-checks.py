"""Independent audit checks for kv-dreaming against the pinned Qwen3-0.6B-Base.

Read-only w.r.t. the project: all outputs go to the scratchpad. Memory-light
(no long full-vocab logits) because the box has ~4 GB free.
"""
import json, sys, time
from pathlib import Path

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, rotate_half

from kv_dreaming.runtime import QwenRuntime, DELIMITERS, STOP_ID
from kv_dreaming.rope import rotate
from kv_dreaming.cache import merge_pair
from kv_dreaming.seeding import calibrate, initialize
from kv_dreaming.runner import RunConfig, run

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
R = QwenRuntime.load(local_files_only=True)
fixture = Path("/home/lenovo/projects/kv-dreaming/data/alice_excerpt.txt")
ids = R.tokenizer.encode(fixture.read_text(), add_special_tokens=False)
res, t0 = {}, time.monotonic()


def hf_cos_sin(positions, like):
    pid = torch.tensor([positions], dtype=torch.float32)
    return R.model.model.rotary_emb(like, pid)


with torch.inference_mode():
    # 1. Parity: custom single-token loop vs HF full forward, bootstrap-first like the real loop.
    seq = [STOP_ID] + ids[:47]
    ref = R.model(torch.tensor([seq]), use_cache=False).logits[0].float()
    st = R.empty()
    act = torch.stack([R.step(t, st) for t in seq])
    res["1_parity"] = {"tokens": len(seq), "max_logit_delta": (act - ref).abs().max().item(),
                       "max_prob_delta": (act.softmax(-1) - ref.softmax(-1)).abs().max().item(),
                       "argmax_equal": bool((act.argmax(-1) == ref.argmax(-1)).all())}

    # 2. kv_dreaming.rope.rotate vs HF rotary (integer, fractional, large positions).
    x = torch.randn(1, 8, 1, 128)
    rope = {}
    for p in [5.0, 5.5, 4097.0, 16447.0, 16447.5]:
        cos, sin = hf_cos_sin([p], x)
        hf, _ = apply_rotary_pos_emb(x, x, cos, sin)
        ours = rotate(x, [p], R.theta)
        back = rotate(hf, [p], R.theta, inverse=True)
        rope[str(p)] = {"fwd_max_abs": (ours - hf).abs().max().item(),
                        "inv_of_hf_max_abs": (back - x).abs().max().item(),
                        "x_max_abs": x.abs().max().item()}
    res["2_rope_vs_hf"] = rope

    # 3. Fractional merge checked by hand with HF cos/sin (independent of rope.rotate).
    A = R.prefill(token_ids=[STOP_ID] + ids[:19])
    B = A.clone()
    ev = merge_pair(B, 5, R.theta)
    worst_k = worst_v = 0.0
    c56, s56 = hf_cos_sin([5.0, 6.0], x)
    c55, s55 = hf_cos_sin([5.5], x)
    for L in range(len(A.keys)):
        k = A.keys[L][:, :, 5:7].float()
        un = k * c56[:, None] - rotate_half(k) * s56[:, None]  # R(-p)
        mk = un.mean(dim=2, keepdim=True)
        exp_k = mk * c55[:, None] + rotate_half(mk) * s55[:, None]
        exp_v = A.values[L][:, :, 5:7].float().mean(dim=2, keepdim=True)
        worst_k = max(worst_k, (B.keys[L][:, :, 5:6] - exp_k).abs().max().item())
        worst_v = max(worst_v, (B.values[L][:, :, 5:6] - exp_v).abs().max().item())
    res["3_fractional_merge"] = {"position": ev["position"], "count": ev["count"],
                                 "positions_after": B.positions[:8], "max_key_err": worst_k,
                                 "max_value_err": worst_v, "length": [A.length, B.length]}

    # 4. Duplicate + log(count) identity end-to-end through runtime.step.
    D = A.clone()
    for L in range(len(D.keys)):
        D.keys[L] = torch.cat((D.keys[L][:, :, :11], D.keys[L][:, :, 10:11], D.keys[L][:, :, 11:]), 2)
        D.values[L] = torch.cat((D.values[L][:, :, :11], D.values[L][:, :, 10:11], D.values[L][:, :, 11:]), 2)
    D.positions = D.positions[:11] + [D.positions[10]] + D.positions[11:]
    D.counts = D.counts[:11] + [1] + D.counts[11:]
    D.validate(finite=True)
    M = D.clone()
    merge_pair(M, 10, R.theta)
    tok = ids[19]
    ld, lm = R.step(tok, D.clone()), R.step(tok, M.clone())
    la = R.step(tok, A.clone())
    res["4_dup_logcount"] = {"dup_vs_merged_max_logit_delta": (ld - lm).abs().max().item(),
                             "dup_vs_undup_max_logit_delta_(should_be_nonzero)": (ld - la).abs().max().item()}

    # 5. Random-KV / soft init scale vs real keys and values.
    cal = calibrate(R, fixture, token_count=1024)
    real = R.prefill(token_ids=ids[:1024])
    rnd = initialize(R, "random", seed=11, prefix_length=64, calibration=cal)
    soft = initialize(R, "soft", seed=11, prefix_length=64, calibration=cal)
    per_layer = []
    for L in range(len(real.keys)):
        rk = real.keys[L][:, :, 4:].float().norm(dim=-1).mean(dim=-1)[0]
        rv = real.values[L][:, :, 4:].float().norm(dim=-1).mean(dim=-1)[0]
        per_layer.append({
            "layer": L,
            "rand_key_norm_ratio_by_head": (rnd.keys[L].float().norm(dim=-1).mean(-1)[0] / rk).tolist(),
            "rand_val_norm_ratio_mean": (rnd.values[L].float().norm(dim=-1).mean(-1)[0] / rv).mean().item(),
            "soft_key_norm_ratio_mean": (soft.keys[L].float().norm(dim=-1).mean(-1)[0] / rk).mean().item(),
            "soft_val_norm_ratio_mean": (soft.values[L].float().norm(dim=-1).mean(-1)[0] / rv).mean().item()})
    kr = torch.tensor([r["rand_key_norm_ratio_by_head"] for r in per_layer])
    res["5_init_scale"] = {"rand_key_ratio_min_max": [kr.min().item(), kr.max().item()],
                           "rand_val_ratio_min_max": [min(r["rand_val_norm_ratio_mean"] for r in per_layer),
                                                      max(r["rand_val_norm_ratio_mean"] for r in per_layer)],
                           "soft_key_ratio_min_max": [min(r["soft_key_norm_ratio_mean"] for r in per_layer),
                                                      max(r["soft_key_norm_ratio_mean"] for r in per_layer)],
                           "soft_val_ratio_min_max": [min(r["soft_val_norm_ratio_mean"] for r in per_layer),
                                                      max(r["soft_val_norm_ratio_mean"] for r in per_layer)],
                           "rand_positions": [rnd.positions[0], rnd.positions[-1]],
                           "bootstrap_position": rnd.bootstrap_position}
    (OUT / "init-scale-per-layer.json").write_text(json.dumps(per_layer, indent=1))
    del real

ident = {"audit": "claude"}

# 6. Determinism + common random numbers (no reduction within 24 tokens).
def go(name, **kw):
    cfg = RunConfig(max_tokens=24, max_seconds=600, snapshot_every=0, **kw)
    st = initialize(R, cfg.initial_state, seed=cfg.initialization_seed, prefix_length=cfg.prefix_length,
                    calibration=cal, alpha_k=cfg.alpha_k, alpha_v=cfg.alpha_v)
    s = run(R, st, cfg, OUT / name, identity=ident)
    return json.loads((OUT / name / "tokens.json").read_text()), s

a, sa = go("rand-evict-a", initial_state="random", policy="evict")
b, _ = go("rand-evict-b", initial_state="random", policy="evict")
c, _ = go("rand-merge", initial_state="random", policy="merge")
e, _ = go("empty-evict", initial_state="empty", policy="evict")
res["6_determinism_crn"] = {"rerun_identical": a == b, "evict_vs_merge_identical_pre_reduction": a == c,
                            "random_vs_empty_first_divergence": next((i for i, (p, q) in enumerate(zip(a, e)) if p != q), None),
                            "stop_reason": sa["stop_reason"], "sec_per_token": sa["elapsed_seconds"] / max(1, sa["generated_tokens"])}

# 7. Forced reductions: bookkeeping + merge-event log size.
for pol in ("evict", "merge"):
    cfg = RunConfig(initial_state="random", policy=pol, budget=80, recent=16, max_tokens=64,
                    max_seconds=600, snapshot_every=0)
    st = initialize(R, "random", seed=11, prefix_length=64, calibration=cal)
    s = run(R, st, cfg, OUT / f"forced-{pol}", identity=ident)
    lines = (OUT / f"forced-{pol}" / "steps.jsonl").read_text().splitlines()
    recs = [json.loads(l) for l in lines]
    nev = sum(len(r["events"]) for r in recs)
    fin = torch.load(OUT / f"forced-{pol}" / "final-state.pt", weights_only=True)["state"]
    res[f"7_forced_{pol}"] = {"summary": s, "events": nev,
                              "bytes_per_event": (sum(len(l) for l in lines) / max(1, nev)),
                              "max_cache_len": max(r["cache_length"] for r in recs),
                              "bootstrap_kept": 64.0 in fin["positions"],
                              "first4_kept": all(float(p) in fin["positions"] for p in range(4)),
                              "sum_counts_plus_evicted": sum(fin["counts"]),
                              "fractional_positions": sum(1 for p in fin["positions"] if p != int(p))}

res["elapsed_seconds"] = time.monotonic() - t0
(OUT / "audit-results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
