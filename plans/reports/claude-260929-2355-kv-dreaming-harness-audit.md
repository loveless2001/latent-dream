# kv-dreaming harness audit (claude, 2026-09-29 23:55)

Scope: read-only review of `/home/lenovo/projects/kv-dreaming/src/kv_dreaming/*` against `260929-kv-dreaming-design.md`. Independent checks were run on the pinned `Qwen/Qwen3-0.6B-Base@da87bfb` (float32, CPU, 4 threads).
- Script and raw results: `claude-260929-2355-kv-dreaming-harness-audit/`
- Runtime: 79 s; outputs written to scratchpad only.

## Verdict

- **PASS:** the 128-token smoke can run now.
- **Before the 2k matrix:** decide F1 and fix F4.
- **Before the 16k extension:** fix F2 and F3, and plan for F5.

## Independent checks (all pass)

| # | Check | Result |
|---|---|---|
| 1 | Custom loop vs HF full forward, 48 tokens, bootstrap first | max logit Δ 1.6e-4, max prob Δ 8.9e-6, argmax 48/48 equal |
| 2 | Stop/mask | By code: 4 IDs set to −inf, stop set disjoint, bootstrap fed not sampled, loop breaks only on a sampled 151643. Codex tests 14/14 |
| 3 | Fractional merge recomputed by hand with **HF's own cos/sin** | positions 5,6 → 5.5, count 2; key err 2.9e-6, value err 0 |
| 4 | Duplicate entry vs merged (count 2, log 2 bias), end to end through `runtime.step` | logit Δ 7.1e-5 (dup vs no-dup Δ = 2.5, so the test is sensitive) |
| 5 | Shared random draws (CRN) + determinism | Rerun identical. Evict ≡ merge before the first reduction. Random vs empty diverge at token 0 |
| 6 | Forced reductions (budget 80, recent 16, 64 tokens) | 48 events per policy; cache ≤ 80; first 4 + bootstrap kept; merge counts sum 128 = every token fed; 48 fractional positions |
| 7 | Init scale vs real cache | Random values 1.01–1.14× real. Soft keys 0.87–1.13×. Soft values 0.61–1.14×. **Random keys: see F1** |

## Findings

- **F1 — decide before the 2k matrix.** Random **key** norms are 0.28–1.64× real per head (median 0.83). 33 of 224 heads are below 0.5× and 85 are below 0.75×, concentrated in layers 0–7.
  - Cause: real pre-norm keys are anisotropic, so `k_norm(randn)` matches the per-channel gain but not real channel energy. Claude's own earlier suggestion was wrong here.
  - Effect: the random prefix is systematically weaker than a real past, especially in early layers.
  - Fix (recommended): calibrate keys the same way as values. Unrotate the real cache keys at their positions, take per-layer/head/channel mean+std excluding the first 4, sample, then RoPE at positions 0..63. Expect a ratio of about 1.
- **F2 — fix before 16k.**
  - `rope.rotate` uses `theta**(-i/d)`; HF uses `1/theta**(i/d)`. inv_freq relative difference is ≤1.2e-7.
  - Resulting mismatch vs HF rotation: 4e-4 at position 4097, 2e-3 at 16447 (|x|max 3.25). Negligible at 2k (max position 2112).
  - Fix: use `model.model.rotary_emb.inv_freq` in `rope.rotate`.
- **F3 — fix before 16k.**
  - Merge events log 16 KB each, vs 0.9 KB for evict. That's ~26 MB per 2k merge run (acceptable) and ~260 MB per 16k run.
  - Fix: log a per-layer summary (mean/min/max of merged/input norm ratio). Put full norms in a side .npy if needed.
- **F4 — fix before running the diagnostic.**
  - `continuation_kl` builds full-vocab logits for all 1,152 tokens: ~700 MB fp32 on top of the 2.4 GB model. The box has 6.8 GB total and ~2 GB headroom with the model loaded, so this is a plausible crash cause.
  - Fix: `logits_to_keep=continuation+1` and take the first 128 rows.
- **F5 — planning, not code.**
  - Speed is 0.18 s/token at a small cache. A 2k run takes ~6–10 min, so the 36-run 2k matrix is ~4–6 h sequential; the 16k extension is ~36 × 1–1.5 h.
  - Needs a GPU backend or a subset.
  - Never run two model processes at once on this box.
- **F6 — interpretation note.**
  - With evict, 60 of the 64 random prefix slots are dropped by ~step 508. Protected slots 0–3 and the bootstrap stay; corrected by codex, msg 20598.
  - With merge, more of the original seed state survives, blurred, for the whole run.
  - So on random starts, the evict-vs-merge comparison partly means "less seed kept vs more seed kept".
  - Generated cache entries also carry earlier seed influence.

## Unresolved questions

- F1: switch random keys to calibrated stats (recommended), or keep `k_norm` as specified and document the under-scale?
- Where should the 16k extension run: local 3060 (needs a CUDA torch build), Modal, or a subset?
