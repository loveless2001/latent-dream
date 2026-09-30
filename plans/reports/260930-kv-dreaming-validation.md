# KV dreaming validation and audit fixes

2026-09-30. Local validation uses the pinned Qwen3-0.6B-Base in float32 on CPU with four threads. Source hashes are recorded with the artifacts. One local model process ran at a time.

## Scope and decisions

The original implementation was authorized in #lab 20582. Claude's [independent audit](claude-260929-2355-kv-dreaming-harness-audit.md) passed the bounded smoke. User message 20597 approved calibrated random keys (F1) and Modal compute with a larger model. Claude owns `kv-dreaming/modal/modal-kv-dream-launcher.py`; Codex owns the harness, tests, and documentation. The main model preset is Qwen3-4B-Base at `906bfd4b4dc7f14ee4320094d8b41684abff8539`. Its independent CUDA audit, model-specific calibration, and measured bounded run precede the matrix.

## Implemented fixes

- F1: schema 2 stores per-layer/head/channel unrotated real-key and value statistics, excluding the first four fixture slots. Random keys sample calibrated coordinates, then rotate, without another `k_norm`. Old calibration files are rejected; checkpoint/backend identity and tensor shapes are checked.
- F2: runtime initialization, calibration, and merging use the model's native `rotary_emb.inv_freq`. Standalone theta compatibility follows HF's reciprocal-power ordering.
- F3: merge events contain per-layer mean/min/max norm summaries, including the merged/count-weighted-input norm ratio. A measured 28-layer event occupies 9,515 JSON bytes, still about 150 MB over 16k reductions; this is a reduction in log size, not negligible storage.
- F4: the real-text diagnostic requests only `continuation + 1` vocabulary-logit rows, then uses the first `continuation`. Tests compare the retained rows to a full forward and check zero KL without reduction, including a one-token continuation.
- Sampling always uses CPU arithmetic. CUDA model logits can still differ from CPU, so generated token identity across devices is not promised. The CLI requests highest float32 matmul precision and deterministic algorithms.
- Added explicit 0.6B and 4B model presets; custom model overrides require a revision. Calibration remains separate for each model/backend.

## Local evidence

All paths below are relative to `kv-dreaming/`.

| Check | Result | Artifact |
| --- | --- | --- |
| Recovered pre-crash mechanics | 14 passed | `artifacts/mechanics.xml` |
| F4 regression checks | 16 passed | `artifacts/mechanics-post-audit.xml` |
| Final mechanics and calibration tests | 20 passed | `artifacts/mechanics-calibrated-v2.xml` |
| Pinned pretrained parity, before F1–F3 | Max logit difference 0.000185013; probability difference 0.000014782 | `artifacts/pretrained-verification.json` |
| Dense-masked eviction parity | Probability difference 6.8545e-7; argmax equal | Same pretrained artifact |
| Schema 2 calibration | 1,024 fixture tokens; leading four excluded | `artifacts/calibration-v2.pt` and `.json` |
| Random/real mean key-norm ratios, seed 11 | Min 0.960367, median 0.998840, max 1.030933 over 224 heads | `artifacts/calibrated-verification.json` |
| Native RoPE parity at 4097, 16447, 30000.5 | Max error 0 against the model's own rotation | Same calibrated artifact |
| Original native-k_norm smoke | 128 tokens, token limit, 27.965 s; fragmented prose | `runs/smoke-random-evict/` |
| Calibrated-key smoke | 128 tokens, token limit, 31.197 s; repetitive punctuation | `runs/smoke-random-evict-v2/` |
| Real-text continuation KL(full || policy) | Evict 0.0338894; merge 0.0541150 | `artifacts/real-text-kl-v2.json` |

Both smoke runs ended with 192 cache slots and zero reductions. They establish bounded execution, logging, and initialization mechanics; they do not compare the cache policies. Their texts, token IDs, uniform draws, manifests, and snapshots are preserved without cleanup. The original calibration and smoke remain pre-F1 evidence and are not silently reused by schema 2.

The diagnostic completed within its 900-second cap: 771.35 s inside the diagnostic, 13:08.94 for the command including model loading. Peak resident memory was 3,808,100 KiB (about 3.63 GiB). It used a 1,024-token real-text prefix and 128 continuation predictions, with budget 512 and recent region 256. Each policy performed 639 reductions and ended with 512 slots. All 128 per-token KL values per policy are finite; recomputed means match the report, and source hashes match the calibrated smoke. Eviction has lower mean distortion on this fixture; this is descriptive evidence from one fixture, not a general ranking of memory policies or generation quality.

The new calibration SHA256 is `f679a8e348099db7a855adcc40746cb987e54f65d87157bbf52ba2d3e092c8cd`. Its fixture SHA256 is `876cf85e4ee35987e57d157bab6584ab093fa0985786aaa212a86492a58049e2`. Exact per-file implementation hashes are in the calibration metadata and calibrated verification artifact.

## Commands

```bash
uv run pytest -q --junitxml=artifacts/mechanics-post-audit.xml
uv run python scripts/verify_pretrained.py --output artifacts/pretrained-verification.json
uv run kv-dream calibrate --local-files-only --output artifacts/calibration.pt
uv run kv-dream run --local-files-only --calibration artifacts/calibration.pt --initial-state random --policy evict --max-tokens 128 --max-seconds 300 --output runs/smoke-random-evict
# Following the user-approved initialization change:
uv run pytest -q --junitxml=artifacts/mechanics-calibrated-v2.xml
uv run kv-dream calibrate --local-files-only --output artifacts/calibration-v2.pt
uv run python scripts/verify_calibrated.py --calibration artifacts/calibration-v2.pt --output artifacts/calibrated-verification.json --smoke-output runs/smoke-random-evict-v2
uv run kv-dream diagnostic --local-files-only --output artifacts/real-text-kl-v2.json --max-seconds 900
```

Calibration, smoke, and diagnostic stdout/stderr and `/usr/bin/time -v` resource logs are retained under `artifacts/`. The original smoke command refers to the old source version recorded in its manifest; current source correctly rejects that legacy calibration.

## Claim limits and next gate

Matching mean key norms does not make independent Gaussian keys/values a consistent model history, and the repetitive calibrated smoke is not evidence of improved generation quality. The KL diagnostic characterizes distortion on one fixed teacher-forced fixture, not dream quality or general semantic preservation.

Correction to audit F6: both policies permanently retain synthetic positions 0–3 and bootstrap position 64. Eviction removes the other 60 original synthetic slots by approximately the 508th generated step, rather than removing the whole prefix. Merge retains approximate contributions; generated cache entries can also carry earlier seed influence.

Local CUDA behavior, 4B parity, and matrix cost are not established by these CPU checks. Claude owns those checks and the bounded timing measurement. No 2k/16k matrix has been launched by Codex during this validation. Local validation is complete, and the source files remain stable for the independent audit.

Subsequent update, 2026-09-30: Claude completed the 4B/CUDA gate and 36-run matrix. Codex's [artifact crosscheck](260930-kv-dreaming-matrix-crosscheck.md) confirms 34 generated-EOS stops and two token limits, but finds one policy pair diverging before any reduction. The saved descriptive results remain usable; cross-container bitwise determinism and causal policy comparisons are not established. No new inference was launched for this crosscheck.

## Changed files

Harness changes are in `kv-dreaming/src/kv_dreaming/{seeding,rope,cache,runtime,sampling,runner,diagnostic,cli}.py`. Regression coverage is in `tests/test_calibration.py`, `tests/test_diagnostic.py`, and the updated mechanics/runner tests. `scripts/verify_calibrated.py` preserves the real-model initialization and smoke evidence. README and the original design's dated amendment describe the approved changes. The Modal launcher belongs to Claude and was not edited by Codex.
