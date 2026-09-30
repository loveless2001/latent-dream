# Fragment start with a BOS prefix: preregistration

Frozen before sink-batch execution, 2026-09-30. User authorization: #lab 20637; implementation specification: 20638; operational P1/P3 definitions proposed in 20639 and accepted in 20640. Codex owns the harness and local checks. Claude owns the independent audit, separate attention-mass diagnostic, and Modal execution. Target tag: `paired-4b-sink-v1`.

## Intervention and fixed plan

The intervention adds a genuine BOS-only prefill cache slot. Calling the supported Qwen3 runtime with `[151643]` alone supplies the key and value at position 0 for every layer. Copy those tensors unchanged. Select the same four 16-token fragments with the same independent selection/noise RNG streams as the original experiment; undo original RoPE and apply native RoPE at virtual positions 1–64. Beta noise applies to fragment keys in unrotated space and fragment values, never the BOS slot. Prefix length and bootstrap position become 65. Initial protected positions 0–3 contain the BOS slot and fragment 0's first three tokens; metadata records these positions explicitly.

| Setting | Frozen value |
| --- | --- |
| Model | Qwen/Qwen3-4B-Base, revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| Backend | Same float32/CUDA explicit eager path and matching calibration identity |
| Starts | Fragment, BOS slot enabled |
| Initialization seeds | 11, 23, 47, 89 |
| Fragment beta | 0, 0.25 |
| Leader directions | Evict then merge, from one initialized cache in one loaded runtime per group |
| Groups / directions | 8 / 16 |
| Budget / recent tail | 128 / 64 |
| Prefix / bootstrap | 65 / position 65 |
| Sampling stream | 101 |
| Per-direction bounds | 2,048 generated tokens, 900 seconds |
| Fixtures | The same four hash-checked sources, source-prefill cap 512 |

No sampling, EOS, policy, merge, or M4 definition changes. Natural EOS is retained, with no minimum-length forcing. Hard bitwise equality of raw leader/follower logits is required through the forward that triggers the first reduction. For the BOS variant the first reduction follows zero-based sample step 63, and the first potentially policy-affected sample is step 64. The original no-sink boundaries are 64 and 65 respectively. The tracker uses the shifted virtual positions directly; the permanently protected original fragment slots cannot reach an all-original-affected boundary.

This is a compound intervention: it also shifts fragment and bootstrap positions, changes the number of protected fragment tokens from four to three, and moves the first reduction one sample earlier. Improvements cannot be attributed uniquely to an attention-sink mechanism. Prefilled BOS state is an operational definition; sink behavior remains an empirical hypothesis.

## Matched baseline and primary outcomes

Baseline: the sixteen fragment directions of `paired-4b-v1`, matched on initialization seed, beta, leader policy, sampling stream, model/backend/calibration, source files, selected fragment token IDs, and original offsets. The original 32-direction report was independently confirmed in [Claude's audit](claude-260930-0215-kv-dreaming-paired-results-audit.md). Soft/random directions are not part of this comparison. There are eight initial-cache groups; their two leader directions are not independent replicates.

P1 is the number of directions that finish without any reduction event. Ending on the reduction-triggering sample does not count as pre-reduction termination, but still supplies no post-reduction policy sample. Report that separate availability count.

P2 is the number of directions with the existing loop flag: a 128-token window with unique-token fraction below 0.12, checked at stride 32. This is a descriptive low-diversity definition, not a general quality score.

P3 is the number of directions with at least one eligible selected-fragment token trigram wholly within samples from step 0 through the first reduction step, **inclusive**. The triggering sample's logits were computed before cache mutation. If EOS occurs first, use the entire output. Cross-window and cross-fragment trigrams are ineligible. Use the existing exclusion of trigrams shared by all four full source fixtures. A completed direction with fewer than three samples has no opportunity and counts as no hit; retain that opportunity flag. A run error or invalid gate censors the experiment rather than counting as no hit. The boundary is the first cache reduction globally, not each fragment's later individual boundary.

Recomputed baseline counts from the retained token streams:

| Outcome | No-sink baseline | Required BOS result |
| --- | ---: | ---: |
| P1: no reduction reached | 8 / 16 | Fewer than 8 |
| P2: loop flag | 5 / 16 | Fewer than 5 |
| P3: pre-policy fragment-trigram hit | 2 / 16 | More than 2 |

The two P3 hits are the two leader directions of beta 0.25, seed 23, each ending at 19 tokens. The baseline artifact is `kv-dreaming/artifacts/fragment-sink-baseline.json`, SHA-256 `a4883df2b52228aad01f70b251c098a1ee71cab12c985de71d151fa599fe518c`. It binds manifest, summary, token, and step-log hashes per direction. The comparison script hash is `21c64264fb88f617cc14bfe2e121f443bb81dec27275aa0496f9116385e0321f`.

All three strict improvements are required for `PASS_OPERATIONAL_CRITERIA`. With a complete valid experiment, failure of any condition gives `STOP_FRAGMENT_STARTS`: stop this fragment-start line of work under the agreed design. It does not refute every attention-sink hypothesis. Missing, incomplete, or gate-failed directions yield `INCOMPLETE_NO_PRIMARY_VERDICT`; do not silently substitute a smaller denominator or launch an automatic retry.

M4 retains its existing post-fragment-boundary recall and matched-seed baseline rules, including missing opportunities and unequal windows. Its reference pool must match sink mode and prefix length, so sink/no-sink trajectories cannot silently mix. Passing P1–P3 supplies evidence that the initializer more often exposes fragment material; M4 still needs sufficient post-boundary opportunities and cannot identify a causal memory mechanism by itself. No significance or broad superiority claim is preregistered.

## Separate descriptive diagnostic

Claude will independently rebuild both initial cache variants for the same seeds and beta values, run the bootstrap forward, and measure attention mass on slot 0 by layer and head. This requires no harness change. Its results describe whether the BOS slot attracts attention in this setting; they do not alter the fixed primary counts, select favorable settings, or waive the equality gate.

## Implementation and evidence

New CLI flag: `--fragment-sink {none,bos}`, default `none`. An omitted CLI prefix length resolves to 65 only for a BOS fragment start; explicitly incompatible lengths are rejected. Direct Python callers pass `sink="bos"` to `initialize_fragment` and set `PairConfig(fragment_sink="bos", prefix_length=65)`. The paired summary includes `fragment_pre_reduction` with P3 window length, opportunity, per-fragment hit token trigrams, and the directional hit flag.

`kv-dream plan-sink-pairs --output PATH` creates the frozen 8-row plan without inference. The reviewed file is `kv-dreaming/artifacts/fragment-sink-plan-reviewed.json`. Each row is directly compatible with the launcher's existing `pairs` entrypoint, which Claude owns. New model runs remain gated on that independent audit. `scripts/compare_fragment_sink.py --baseline-root PATH --sink-root PATH --output PATH` performs the matched offline P1/P2/P3 comparison; omission of `--sink-root` produces only the baseline artifact. `kv-dream analyze-pairs` continues to produce the within-condition M4 analysis.

Validation: `uv run pytest -q --junitxml=artifacts/mechanics-fragment-bos-final.xml` passed **46 tests in 2.12 seconds**. New coverage includes bitwise BOS-prefill equality at both beta values; unchanged fragment selection and value noise; native-RoPE rephasing at shifted positions; paired native-Qwen equality through step 63 and post-reduction steps 64–65; shifted protected-slot boundaries; P3 inclusive/exclusive window edges; rejection of incompatible configs; sink plan CLI round trips; and mode-separated M4 baselines. The default no-sink cache and metadata match the frozen v1 initializer bitwise in both beta conditions. The original source set is retained at `artifacts/paired-v1-source/`; the regression fixture's original SHA-256 is `042b1b54ada6e4747676e331f5a4bb137f506739b415cfe101ccd66bc6a47218`.

The first test pass had one negative-config assertion hit the tiny test model's position limit before the intended prefix-length check. Bounding that test's token cap corrected the fixture; the implementation's BOS integration already passed. Both test XML artifacts are retained. No sink GPU inference was executed by Codex for this handoff.
