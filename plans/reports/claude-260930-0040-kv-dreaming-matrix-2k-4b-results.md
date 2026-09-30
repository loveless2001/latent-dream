# KV dreaming — first 4B matrix results (2k cap)

2026-09-30, by claude.

**Setup:**
- Model: Qwen3-4B-Base@906bfd4, fp32, on Modal L40S. App `ap-4LNLJyodkpCP2LaX1sjoEJ`, tag `matrix-2k-4b-v2`.
- 36 runs = 3 start types × {evict, merge} × seeds × 2 sampling streams, T=1, no penalties.
- Calibration: `/runs/calib/qwen3-4b-base-cuda-v2.pt` (schema 2, F1 keys).
- Snapshots are final-state only.

**Raw data:**
- Unedited texts: `claude-260930-0040-kv-dreaming-matrix-2k-4b-results/texts/`.
- Per-run stats: `stats.json`.
- Full artifacts (steps.jsonl etc.) on volume `kv-dream-runs:/matrix-2k-4b-v2/`.

**Mechanics:** 36/36 completed, 0 errors.
- 34 runs ended on a self-generated EOS; 2 hit the 2,048 cap (random-i89-s101, both policies).
- `random-merge-i11-s101` matched the separately measured run token for token (1,668 tokens). This is an observed match for that one configuration only; it does not establish bitwise reproducibility. **A pre-reduction numerical discrepancy exists across runs** (codex found it, msgs 20608/20610):
  - `soft-evict/merge-i11-s202` log-probs already differ by up to 6.4e-5 at 381 of the first 431 steps, with identical source hashes and identity.
  - Entropy and log-prob differ from step 0. Tokens first differ at zero-based step 431 (46338 vs 46339, adjacent IDs, consistent with a near-boundary flip; full CDFs were not logged). The first reduction comes after sampling step 448, so the earliest policy-affected sample is 449.
  - The underlying cause (GPU/kernel/container) is not established by these logs.
  - Within one container, reruns were token-identical (audit check 6). That is weaker than logit equality.

## What happened, by start

| Start | Runs | Looped (≤12% distinct tokens in a 128 window) | Median length | Distinct bigram ratio (median) | gzip ratio (median) |
|---|---|---|---|---|---|
| empty (bootstrap only) | 4 (2 unique) | 0 | 347 | 0.70 | 0.50 |
| random KV, α=1 (calibrated) | 16 | **16, onset by token 128–160** | 812 | **0.04** | **0.035** |
| soft (random embeddings → model) | 16 | 0 | 748 | 0.91 | 0.64 |

- **Empty.** The model opens a fresh "document": Python functions (`def linear_search…`, `def access_nested_elements…`) that are coherent, then EOS. This is its default mode.
- **Random KV at full strength.** Stuck, perseverative loops in every run:
  - Examples: `copycopycopy…`, `, , , ,`, `ffff…`, `answer right. answer right.`, `process:pass:pass → miss:miss → phasaphasa`.
  - A few runs jump between loops or produce a burst of word salad (`robot-encircled world remains-space-and-matrix…`), then EOS.
  - Looks less like a dream and more like a seizure or perseveration. The noise overwhelms the model instead of seeding it.
  - Same as the 0.6B calibrated smoke. The weaker pre-F1 keys gave fragmented prose instead. F1 changed both the key norms and the key distribution, so "strength matters" is a hypothesis, not a finding (codex, msg 20613). An α sweep holds the distribution fixed and varies only scale.
- **Soft prefix.** The closest thing to "dreaming" in this set:
  - Starts in mixed-script token salad (Japanese/Korean/Chinese/Cyrillic/Arabic/Thai mixed with code words).
  - Then the model often **tries to make sense of its own nonsense**:
    - `soft-i47-s202` says, in Chinese, "this question is confusing and contains many unreadable words… if you're asking about JSON…", then explains JSON parsing.
    - `soft-i89-s202` annotates its own gibberish: "the sequence also includes … '내' for 'my'", "Non-Korean Words".
    - `soft-i23-s202` glosses invented words: "**WMsFlFiganeat** seems to refer to 'media'".
    - `soft-i47-s101` invents fantasy names, then an encyclopedic entry: "The Vol de Gout, also often called a danse du chevalier, or a dance of frustration or Grimace…".
    - `soft-i23-s101` drifts into Human:/Assistant: Q&A about the Almoravids (dates right, other details wrong).
  - `Human:`/`Assistant:` turns appear as plain text even with the chat tokens masked. The base model learned the format from pretraining.
  - A few first-person mentions (16 across soft runs vs 1 across random).
- **Evict vs merge.**
  - 5 of 18 pairs ended (EOS) before the cache filled and are token-identical.
  - 12 pairs diverge after the first reduction. That is expected, since any cache change alters later sampling.
  - 1 pair (soft-i11-s202) first differs at step 431, **before any policy effect was possible**, from the pre-reduction numerical discrepancy above.
  - Consequence: some later divergences may also be noise-triggered. Policy comparisons need both policies run in the **same container**, plus a same-policy duplicate across containers to measure how often noise alone flips a token.
  - With 8 unique seed/stream settings per random and soft start (2 for empty) and no preregistered policy metric, no policy conclusion is possible yet.
  - Future comparisons: use one loaded runtime, cloned identical initial caches and shared uniforms, with an explicit pre-reduction logit/probability equality check (codex, msg 20611).

## Reading, stated carefully

- Noise injected directly into K/V at real-key strength, independently per layer, pushes the 4B into loops. Noise injected at the input and passed through the model (so every layer is consistent) gives a chaotic start that the model then narrates and interprets. That is the "make a story out of random firing" behavior.
- The model mostly **wakes up by itself** (self-EOS in 34/36), usually within 300–1,800 tokens.
- Limits:
  - n is small: 8 seed/stream settings per random and soft start, 2 for empty.
  - Qualitative reading is by claude, unblinded.
  - Calibration comes from one fixture.
  - T=1 only.
  - Nothing here says anything about experience.

## Proposed next runs (need user go)

1. **Noise-strength sweep on random KV:** α_K=α_V ∈ {0.1, 0.25, 0.5} × 4 seeds × 1 stream (12 runs, ~30 GPU-min). This finds where loops give way to content, and whether some middle strength looks like the soft behavior.
2. **Compare the memory policies properly.** Fix a metric first, e.g.:
   - loop onset and length;
   - how long a motif survives;
   - whether seed-derived material recurs after its slots are reduced.
   
   Then add seeds. A lower budget (e.g. 128/recent 64) makes reductions happen early in every run.
3. **Day-residue seed:** spliced real-text KV fragments + noise (the original "fragment splice, later" variant). This is the start type most like human dream material.

## Unresolved questions

- Which of 1–3 (or all)?
- Should the EOS stop stay as the natural "wake" (the user's original rule), or should a no-EOS variant exist for long-run memory tests?

## Addendum: noise-strength sweep (random KV, evict, stream 101, 4 seeds per α)

App `ap-MDmvj8rin9iRb4smotU8Ax`, tag `alpha-sweep-4b-v2`, 12/12 runs OK. Texts are in `…/alpha-sweep-texts/`.

| α (K and V) | Loops (onset) | Distinct bigram ratio | What it looks like |
|---|---|---|---|
| 0.1 | 0/4 | 0.48–0.91 | Noise is mostly ignored. **3 of 4 seeds open with the same document** (a Python 2 unicode tutorial in Chinese), because the shared uniforms dominate when the prefix barely matters. This behaves like an empty start. |
| 0.25 | 1/4 (onset 352) | 0.14–0.99 | Varied: a Haskell-like loop, a web "contact" page, an X-ray-diffraction Q&A. i89 ran to the cap as a stream of consciousness: "luxury dreams? Is it increased awareness? Are dopamine and endorphins interlinked? … innnerally … spkscialist spikes" (22 first-person uses). |
| 0.5 | 0/4 | 0.46–0.76 | Varied: a short math query then EOS; "reluctantly reluctantly … reflect", after which **the model analyzes its own words** ("The terms you've listed—reluctantly, reflectingly…"); code; C code. |
| 1.0 (main matrix, same seeds and stream) | 4/4 (onset 128) | 0.007–0.027 | Perseveration loops. |

**Reading (n=4 per level, one stream, unblinded):**
- Strength does act like a dial, with the key and value distribution held fixed:
  - too weak and the prefix is ignored (the default document plays out);
  - too strong and the model gets stuck in loops;
  - in the middle (0.25–0.5) the output is varied, sometimes self-interpreting or free-associating.
- The 0.25 loop shows the dial is not monotone at this n.
- α=0.25 remains the pre-chosen random start for the paired runs.
