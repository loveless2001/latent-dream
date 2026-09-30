# Latent Dream

[Live report](https://loveless2001.github.io/latent-dream/) · [GitHub repository](https://github.com/loveless2001/latent-dream)

A frozen Qwen3 model starts from synthetic KV state, receives one bootstrap token, then feeds its own sampled tokens back into inference. No chat template, task prompt, training, or output cleanup is applied. This is a generation experiment; textual behavior does not settle subjective experience.

Working folder: `/home/lenovo/projects/latent-dream/`. The former `kv-dreaming/` path remains a local compatibility symlink so recorded evidence paths and the existing environment continue to resolve. The Python package and CLI retain their `kv_dreaming` and `kv-dream` names.

Open [can language models dream?](index.html) for the standalone HTML report, with featured soft-start configurations, unedited 4B/8B samples, interactive comparisons, and links to the evidence. It works directly from a local file and uses no external assets.

See the [current experiment summary](plans/SUMMARY.md) for the completed 4B campaign, 8B generous-budget survey, and 8B noise sweep: 128 primary trajectories across six batches. Soft starts gave the most consistently varied, non-looping samples in the tested configurations; direct random-KV starts were fragile. The audited **STOP_FRAGMENT_STARTS** verdict stands for the frozen 4B fragment design. The gist-level test, instruct comparison, and 16k-token extension remain unstarted.

All plans, reports, and their local evidence packets are now under [plans/](plans/README.md), with an archived #lab thread and a per-file [consolidation manifest](plans/consolidation-manifest.json). Old report paths remain compatibility symlinks; frozen evidence bytes are preserved. The [historical 4B campaign summary](plans/reports/claude-260930-1007-kv-dreaming-campaign-summary.md) and [independent results audit](plans/reports/claude-260930-0215-kv-dreaming-paired-results-audit.md) retain the earlier phase record.

The agreed [design](plans/reports/260929-kv-dreaming-design.md) defines a 64-slot virtual prefix, 512-slot cache, protected initial/bootstrap/recent positions, and two memory policies: eviction and approximate count-aware KV merging. Merges undo RoPE, average in float32, use a weighted fractional position, rerotate, and add a log-count attention bias. The model's native modules are reused in the same explicit attention path for both policies.

The local mechanical-check preset is `qwen3-0.6b-base`, pinned to `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. The main experiment preset is `qwen3-4b-base`, pinned to `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`. The local environment is CPU-only, pinned through `uv.lock`; the separately owned [Modal launcher](modal/modal-kv-dream-launcher.py) supplies CUDA. Model downloads use the ordinary Hugging Face cache.

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/verify_pretrained.py --output artifacts/pretrained-verification.json
uv run kv-dream calibrate --local-files-only --output artifacts/calibration-v2.pt
uv run kv-dream run --local-files-only --calibration artifacts/calibration-v2.pt --initial-state random --policy evict --max-tokens 128 --max-seconds 300 --output runs/smoke-random-evict-v2
```

The calibration text in `data/` is a fixed public-domain excerpt with source and hash recorded in `data/source.json`. Schema 2 calibration stores per-layer/head/channel means and standard deviations for unrotated real keys and values, excluding the first four slots, plus embedding statistics. Random keys sample those calibrated coordinates without a second normalization, then apply RoPE; soft-prefix states come from an ordinary forward over random embeddings. Legacy calibration files are rejected. Calibrate separately for each model and execution backend: model/revision, dtype, device, and thread count must match. Fixture and calibration hashes are recorded for provenance. All runtime rotation and merge paths use the model's native inverse-frequency buffer.

Only these output IDs are masked: `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`. The bootstrap input and generated EOS both use ID 151643; only a newly generated EOS stops inference. Other added tokens remain possible and are logged. Sampling is full-vocabulary temperature 1, inverse CDF in token-ID order, from a separate reproducible uniform stream. Sampling arithmetic runs on CPU for both backends; this does not guarantee matching tokens when CPU and CUDA model logits differ.

Each fresh output directory contains `manifest.json`, `steps.jsonl`, `tokens.json`, unedited `text.txt`, `summary.json`, and cache snapshots. Merge events include each layer's mean/min/max input and merged key norms, plus the merged-to-count-weighted-input norm ratio across heads. Steps record the uniform draw, raw/post-mask entropy, blocked mass, EOS probability, token log probability, logical position, and physical cache size. Existing directories are never overwritten. Errors and limits are distinct stop reasons. The wall-time check is between forward calls; a running forward is not forcibly interrupted.

Periodic nonterminal snapshots can be resumed into a **new** directory with `--resume PATH`. Checkpoint/backend identity, source hashes, stored cache, cursor, uniforms, and sample history are restored. Final snapshots are terminal; resume from a preceding periodic snapshot. Preserve the original run directory as the provenance of earlier steps. There is no automatic restart or silent extension past the token cap.

After harness review, the descriptive real-text diagnostic is available:

```bash
uv run kv-dream diagnostic --local-files-only --output artifacts/real-text-kl.json --max-seconds 900
uv run kv-dream plan-matrix --output artifacts/initial-matrix.json
```

`plan-matrix` only writes the 36-condition plan (18 unique starts per cache policy); it launches nothing. The initial matrix uses a 2,048-token cap. A later extension can use 16,384 tokens, inside the pinned base model's 32,768-position range. Scaled RoPE, 100k-token runs, and broad token filtering remain outside this implementation. The real-text KL reference retains only 129 full-vocabulary output rows for the 128-token continuation, avoiding a full-prefix logits allocation.

The user approved calibrated random keys and Modal compute in #lab message 20597, then the noise sweep, paired comparison, and fragment followups in message 20614. Claude owns the launcher and independent 4B/CUDA audit. The initial matrix completed, but a [crosscheck](plans/reports/260930-kv-dreaming-matrix-crosscheck.md) found divergence before the first policy reduction in one independently executed pair; cross-container pre-reduction reproducibility is not established. Use `--preset qwen3-4b-base` for the harness directly, or the launcher's `--model 4b`. Explicit harness `--model` overrides must include `--revision`.

The paired followup initializes a cache once and clones it for eviction and merging in the same loaded runtime. The leader generates freely; the follower receives the leader's token history. Both use the same uniform draws. `--leader-policy both` runs eviction as leader, then merging as leader, from the same original cache. Raw logits must be bitwise equal until a previous reduction can affect a forward; a failure saves both logit vectors and stops. KL and token-flip summaries begin on the step after the first reduction. Each direction has its own token and wall-time limits; follower EOS samples are hypothetical and do not stop the leader. Paired runs save terminal states and cannot resume.

```bash
uv run kv-dream plan-pairs --output artifacts/paired-plan.json
uv run kv-dream paired --local-files-only --calibration artifacts/calibration-v2.pt --initial-state fragment --fragment-beta 0.25 --leader-policy both --max-tokens 128 --output runs/fragment-pair-smoke
uv run kv-dream analyze-pairs --input-root runs/paired-matrix --output artifacts/paired-recall.json
```

`plan-pairs` writes 16 groups / 32 leader directions for the 4B preset, budget 128, recent tail 64, prefix 64, initialization seeds 11/23/47/89, sampling seed 101, and cap 2,048. Starts are soft, random with alpha K/V 0.25, and fragments with beta 0/0.25. The example smoke uses the default local 0.6B preset; run plan rows with their declared 4B preset and matching CUDA calibration only after the independent audit.

Fragment sources are four frozen, hash-checked fixtures in [sources.json](data/fragments/sources.json), spanning fiction, code, science, and a political document. Each is forwarded separately through its first 512 tokens; a seeded permutation and offsets select four 16-token blocks. Native RoPE is undone and reapplied at virtual positions 0–63. Values are preserved at beta 0. Noise at beta > 0 is additive `beta * N(calibrated_mean, calibrated_coordinate_std)` in unrotated key/value space. Source IDs, hashes, positions, and token IDs are retained. Recall uses token trigrams within each selected fragment, excluding trigrams shared by all four full source fixtures; cross-fragment trigrams are excluded. The offline baseline uses other initialization seeds with matching conditions and excludes any comparator seeded with an eligible target trigram. Nulls, comparator exclusions, and available output lengths remain explicit. The full [paired/fragment contract](plans/reports/260930-kv-dreaming-paired-fragment-contract.md) defines boundaries and claim limits.

The original paired experiment completed and its [results](plans/reports/260930-kv-dreaming-paired-results.md) were [independently confirmed](plans/reports/claude-260930-0215-kv-dreaming-paired-results-audit.md). The user approved a BOS-prefix followup in #lab 20637. `--fragment-sink bos` copies the genuine BOS-only prefill cache at slot 0, shifts the same selected fragments to positions 1–64, and puts the bootstrap at 65. Noise affects only fragments. Protected positions 0–3 now cover the BOS slot and the first three fragment tokens. The CLI infers prefix length 65 when omitted; an explicit incompatible length is rejected. The default `--fragment-sink none` preserves the original initializer bitwise in regression checks.

```bash
uv run kv-dream plan-sink-pairs --output artifacts/fragment-sink-plan-reviewed.json
uv run python scripts/compare_fragment_sink.py --baseline-root plans/reports/260930-kv-dreaming-paired-results/raw --sink-root runs/paired-4b-sink-v1 --output artifacts/fragment-sink-comparison.json
```

The new plan fixes eight groups / sixteen directions, beta 0/0.25, the same four seeds, budget 128, recent 64, stream 101, and a 2,048-token cap. No inference is launched by plan generation or comparison. The [preregistration](plans/reports/260930-kv-dreaming-fragment-sink-preregistration.md) fixes P1 fewer pre-reduction stops, P2 fewer loop flags, and P3 more directions with a selected-fragment trigram before policy can affect a sample. The no-sink counts are 8, 5, and 2 out of 16. All three must improve; incomplete experiments get no primary verdict. A negative result stops fragment starts under this design. Successful BOS insertion does not itself establish attention-sink behavior; Claude owns a separate descriptive attention-mass diagnostic, audit, and GPU execution.

The [completed BOS followup](plans/reports/260930-kv-dreaming-fragment-sink-results.md) has all sixteen directions, 1,024 passing equality gates, and no artifact discrepancies. P1 improves 8→0 and P2 improves 5→2, but P3 remains 2→2, giving the preregistered **STOP_FRAGMENT_STARTS** verdict. Post-policy availability increases from eight to sixteen directions. M4 has nine positive boundary measurements across three fragment/direction combinations, all seed 11 / beta 0.25; these repeated measurements are not independent successes. The separate bootstrap diagnostic shows concentrated slot-zero attention with BOS, without changing the primary verdict. The report retains raw artifacts, source hashes, and a separate replay of all 34,200 reduction events. Claude's [independent audit addendum](plans/reports/claude-260930-0215-kv-dreaming-paired-results-audit.md) confirms P1–P3, the 1,024 exact gates, and the stop verdict.

Interpretation: both policies retain initial positions 0–3 and the bootstrap. Without the BOS slot, eviction eventually drops the other 60 original synthetic slots. With it, 61 of the 64 fragment slots are unprotected. Merge retains approximate contributions; generated cache entries can also carry seed influence. The policies therefore differ in retention of the initial seed as well as later context.

Implementation references were inspected in `../memory/memsub/`: logical/physical separation, rotation helpers, and dense-masked continuation checks. There is no import or runtime dependency on that checkout. Its older merger is a different algorithm; this merger is approximate and makes no claim to preserve semantics better than eviction.

Status and artifact evidence: [validation report](plans/reports/260930-kv-dreaming-validation.md). Original pre-F1 artifacts are retained; use a fresh output path for every new run.
