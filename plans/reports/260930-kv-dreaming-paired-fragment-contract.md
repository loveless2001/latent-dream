# KV dreaming paired policy and fragment contract

Prepared 2026-09-30 before paired 4B execution. User authorization: #lab 20614; scope agreed with Claude in 20615–20620. Codex implements and checks the harness; Claude owns `kv-dreaming/modal/modal-kv-dream-launcher.py`, independent CUDA audit, and execution. The noise sweep is a separate experiment. Its outcomes do not change the preselected random alpha 0.25 here.

## Fixed plan and execution

`kv-dream plan-pairs` writes 16 group rows, each with `leader_policy=both`, for 32 directional comparisons. Each row can be translated directly to the `paired` CLI using underscores as hyphens. Model preset: `qwen3-4b-base`, pinned revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`. Backend and calibration must match exactly.

| Setting | Fixed value |
| --- | --- |
| Starts | soft; random alpha K/V = 0.25; fragment beta = 0; fragment beta = 0.25 |
| Initialization seeds | 11, 23, 47, 89 |
| Sampling seed | 101 |
| Prefix / physical budget / recent tail | 64 / 128 / 64 |
| Per-direction limits | 2,048 generated tokens; 900 seconds measured between paired steps |
| Protected slots | Initial virtual positions 0–3, bootstrap, recent tail |
| Direction order | Eviction leader, then merge leader |

One model runtime and one initialized prefix are used per group. Each direction clones that prefix into leader and follower states; the original state hash must remain unchanged. The follower always receives the leader's preceding output token. Both candidate samples use the same recorded uniform draw and the existing full-vocabulary temperature-1 sampler with the same four masked delimiters. Only leader EOS stops a direction. No tuning, automatic retries, cap extensions, or resumed paired runs are implicit in this plan.

The equality gate compares the raw logit bytes on every forward that has no prior reduction. This includes the sample step that triggers the first reduction: reducing its newly extended cache cannot have affected the logits already computed. A mismatch preserves `failure.json`, `failure-logits.pt`, partial raw outputs, summary, and terminal states when valid, then raises. The gate is never relaxed to a tolerance. With a 64-slot prefix and budget 128, the first reduction is after zero-based sample step 64 and the first potentially policy-affected sample is step 65, provided EOS does not occur first.

Each direction records `manifest.json`, `uniforms.json`, `steps.jsonl`, `tokens.json`, unedited `text.txt`, `summary.json`, and `final-state.pt`. The manifest includes checkpoint/backend identity, calibration hash, source hashes, initial-cache hash, and fragment provenance. Step records include both distributions' entropy/sampling summaries, hypothetical follower token, KL, flip, gate, reduction events, and physical lengths. Existing output paths are refused. EOS and token cap are completed bounded observations; time limits, interruption, and errors are incomplete and must be reported separately. A forward is not forcibly interrupted by the wall-time limit.

## Metrics fixed before data

M1 is the arithmetic mean of `KL(leader || follower)` over post-reduction steps, using masked distributions on identical leader history. Log-softmax follows the sampler's float32 path; normalization and KL summation use float64. M2 is the fraction of these steps where candidate token IDs differ under the shared uniform draw. Both are null if there are no post-reduction observations. Per-step values include the earlier gate period for inspection.

M3 describes the leader trajectory: first 128-token window with unique-token fraction below 0.12, scanned at stride 32; EOS step; and distinct-bigram fraction. Loop onset is the number of generated tokens available at the end of that window. Other step indices are zero-based. These are descriptive behavior measures, not quality judgments. The follower entropy is recorded per step; its candidate tokens are not a free-running trajectory.

M4 measures lexical recurrence per selected fragment and leader trajectory. Eligible token trigrams are contained wholly within each 16-token fragment, deduplicated, and exclude the intersection of trigrams found in all four **full frozen fixture texts** under the pinned tokenizer. The denominator is the count of eligible unique trigrams. Crossing a splice boundary does not create an eligible trigram.

For each policy, provenance tracks original virtual slots through physical selection and merge unions. Three boundaries are retained for each fragment: first original slot touched by a reduction; all unprotected original slots touched; all original slots touched. A merge touches its inputs but does not imply their information is lost. The first four original positions are permanently protected, so the last boundary for the fragment containing them is normally null. Reduction at step `t` precedes the measurement window: only trigrams wholly inside generated tokens with indices greater than `t` count. Each boundary reports recall, every eligible trigram's last recurrence, the last recurrence end step overall, and available output length. Unreached boundary, empty eligible set, or fewer than three later tokens gives null, not zero.

`kv-dream analyze-pairs` computes a descriptive baseline from retained outputs without model execution. For each target fragment and boundary, comparators must have a different initialization seed but the same leader direction, fragment beta, budget, recent tail, token cap, sampling seed, model/backend identity, harness source hashes, calibration hash, and source-manifest hash. They must complete at EOS or token cap. Exclude any comparator whose selected fragment blocks contain any eligible target trigram. Measure the remaining outputs against the target eligible trigrams starting after the **target's same absolute boundary step**. Report every exclusion, eligible comparator count, available window length, mean comparator recall, and observed recall minus this mean. Missing comparator or opportunity produces null. Unequal output lengths remain visible; this is neither a significance test nor a causal-memory estimate.

## Fragment construction and sources

The manifest `kv-dreaming/data/fragments/sources.json` fixes four distinct sources and four genres. Every source is forwarded independently using its first 512 tokens, with no template or special tokens inserted. A CPU RNG seeded by the initialization seed permutes all four sources and selects offsets uniformly from valid 16-token windows. Source selection is separate from noise draws, so beta changes neither permutation nor offset. Keys are unrotated at their original positions and rerotated at virtual positions 0–63 using native checkpoint inverse frequencies. Beta 0 preserves selected values exactly. For beta greater than zero, keys in unrotated space and values receive additive `beta * N(calibrated coordinate mean, calibrated coordinate std)` before key rerotation, with no extra normalization. The noise RNG uses seed XOR `0x4B564452`.

| ID / genre | Frozen source | SHA-256 |
| --- | --- | --- |
| alice / fiction | Byte-for-byte existing [Gutenberg Alice fixture](https://www.gutenberg.org/cache/epub/11/pg11.txt) | `876cf85e4ee35987e57d157bab6584ab093fa0985786aaa212a86492a58049e2` |
| sampling / Python code | Frozen copy of local `src/kv_dreaming/sampling.py` | `e1498f6d54d1bbd5bcbeb01ae5ea291aa35015525a915eb90dccb00d859a68fa` |
| darwin / science | 8,000 characters beginning “When on board H.M.S.” in [Origin of Species](https://www.gutenberg.org/cache/epub/1228/pg1228.txt) | `1e4934287d0ed0b2a9c32c785f2dc24dc2f19f6b05271207f9a3726853bfea83` |
| declaration / political document | [National Archives transcription](https://www.archives.gov/founding-docs/declaration-transcript), normalized text from the opening through “our sacred Honor.” | `25834be8d0ef95ebdb22253c9535e0fd910a1574bda1110406a5498a4e60be00` |

Source bytes are hash-checked before inference. The runtime manifest retains source metadata, full and prefill token IDs, selected token IDs, original offsets, virtual positions, eligible/excluded trigrams, beta, and initialization seed. The prefills carry preceding context from each individual source; rephasing does not erase that context. Exact lexical recurrence can arise from model prior knowledge, earlier generated text, or retained/protected state. It does not establish episodic recollection, semantic fidelity, subjective experience, or general superiority of merging.

## CLI and validation handoff

New commands: `plan-pairs --output PATH`; `paired --leader-policy {evict,merge,both}`; `analyze-pairs --input-root PATH --output PATH`. Paired accepts the shared model/calibration/limit/seed flags plus `--initial-state fragment`, `--fragment-sources PATH`, and `--fragment-beta FLOAT`. `--snapshot-every` must be zero. For a group row, use its declared preset and condition flags, add the target backend and model-specific calibration, and provide a fresh output directory. The launcher may serialize the flat plan rows; no internal Python function call is required.

Local tests cover forced pre-reduction failure and saved logits; equality through the triggering reduction step; leader-history teacher forcing despite hypothetical follower EOS; both directions sharing an unchanged prefix; native tiny Qwen attention through actual cache reductions; analytic masked KL; source hash rejection; native-RoPE fragment rephasing and unchanged beta-0 values; reproducible noise with unchanged fragment selection; pinned-slot boundaries; exclusion of boundary-crossing trigrams; comparator contamination, nulls, and baseline arithmetic; and plan-to-CLI round trips. These checks validate mechanics on CPU, not 4B/CUDA reproducibility or research outcomes. Claude's independent audit is required before the paired matrix.

Validation: `uv run pytest -q --junitxml=artifacts/mechanics-paired-fragments-native.xml` passed **31 tests in 1.67 seconds**. The XML is retained under `kv-dreaming/artifacts/`. `artifacts/paired-plan-reviewed.json` is the final generated plan and records the current harness and fixture-manifest hashes; the earlier `paired-plan.json` is a retained draft. The initial failed test artifact is also retained: its sole failure was a Python tuple/list metadata representation mismatch, corrected to JSON-native lists before the passing runs. No paired 4B execution was performed by Codex for this handoff.
