# KV dreaming — campaign summary (2026-09-29 → 30)

The 4B campaign closed at the user's request in #lab 20649. The independently confirmed **STOP_FRAGMENT_STARTS** verdict stands for its frozen design. The user subsequently requested an 8B test with a more generous budget from Claude (#lab 20652); that separate follow-up does not change the completed 4B results.

## Idea (the user's)

Let a frozen LLM "dream":
- start from a synthetic KV cache instead of a prompt;
- feed its own output back in, with chat/think tags masked;
- keep memory bounded by compressing or evicting the cache;
- run until the model emits EOS ("wakes") or hits a hard limit.

Nothing is trained; this only observes what the process does.

## Where things are

- **Code:** `/home/lenovo/projects/kv-dreaming/`. Codex owns the harness (`src/kv_dreaming/`, tests, `artifacts/`). Claude owns only `modal/modal-kv-dream-launcher.py` and `modal/runs-index/`. It is **not a git repo yet**.
- **Reports:** `/home/lenovo/projects/plans/reports/`.
  - Design: `260929-kv-dreaming-design.md`.
  - Validation: `260930-kv-dreaming-validation.md`.
  - Harness audit: `claude-260929-2355-kv-dreaming-harness-audit.md`.
  - Matrix and sweep: `claude-260930-0040-kv-dreaming-matrix-2k-4b-results.md` + `texts/` + `alpha-sweep-texts/`.
  - Paired: `260930-kv-dreaming-paired-results.md`.
  - Sink: `260930-kv-dreaming-fragment-sink-results.md`.
  - Audits of both: `claude-260930-0215-kv-dreaming-paired-results-audit.md` + folder.
  - Preregistration: `260930-kv-dreaming-fragment-sink-preregistration.md`.
- **Modal volumes:** `kv-dream-runs` (all run outputs and the 4B calibration `/calib/qwen3-4b-base-cuda-v2.pt`) and `kv-dream-hf-cache` (Qwen3-4B-Base weights).

## Setup

- Qwen3-4B-Base@906bfd4, fp32 on Modal L40S. The 0.6B-Base is used locally for mechanics only.
- T=1, no penalties, a shared uniform stream, masking of `<|im_start|> <|im_end|> <think> </think>`, and stop only on self-sampled `<|endoftext|>`.
- Cache: 64-slot synthetic prefix (65 with the added BOS slot), then bounded to 512 (matrix) or 128 (paired).
- Two memory policies:
  - **evict** (drop the oldest);
  - **merge** (average adjacent entries in unrotated space, with fractional position and a log(count) attention bias).

## Results

| Experiment | Finding |
|---|---|
| Harness | CUDA parity 8e-5, RoPE exact, merge maths exact. Pre-reduction bitwise gate held in **2,664/2,664** checks across paired batches. Cross-container GPU runs are *not* bitwise identical (log-prob drift ~6e-5), so policy comparisons run in one runtime. |
| Start types (36 runs, 2k) | The model **wakes by itself** (self-EOS) in 34/36. **Empty** start: ordinary code documents. **Random KV (α=1):** 16/16 perseveration loops (`copycopycopy`, `ffff`). **Soft** start (random embeddings through the model): 0/16 loops, the most dream-like, with multilingual gibberish that the model then *tries to interpret*. |
| Noise dial (α sweep) | α=0.1: noise ignored (the default document plays out). α=0.25–0.5: varied, self-interpreting, stream of consciousness ("luxury dreams? … dopamine and endorphins interlinked…"). α=1: loops. |
| Evict vs merge (paired, shadow design) | After memory is reduced, the policies give clearly different next-token predictions (KL 0.04–0.56 nats; 6–50% token flips under the same draw). There is no consistent direction, and n is small. |
| Attention at slot zero | At the bootstrap forward, random and no-BOS fragment caches give slot zero little attention (~0.002–0.004). Adding BOS to fragments raises it to ~0.75–0.77, similar to the soft start (~0.75). This diagnostic does not measure sinks at other positions or later steps, or establish that a sink is necessary. |
| "Day residue" fragment starts | Adding BOS reduces directions ending before reduction from 8→0/16 and loop flags from 5→2/16. P3, eligible fragment-trigram hits through the first reduction, stays 2→2/16. Preregistered verdict: **STOP_FRAGMENT_STARTS**. Some later verbatim recurrence exists: M4 has nine positive boundary measurements across three fragment/direction combinations, all seed 11 / beta 0.25; these are not nine independent successes. Possible gist/genre echoes (a made-up Gutenberg Dickens text, "biologist and explorer Robert Boyle") are unblinded anecdotes; source attribution was not tested. |

## What it adds up to

- A frozen base LLM **can generate without a user-written task prompt**: supplied random/soft-prefix or fragment KV state plus a bootstrap token produces self-generated streams, often ending in self-sampled EOS. Fragment caches still contain information from external source text.
- Whether this looks dream-like depends on **how the seed enters**:
  - noise that goes *through* the model, or noise injected at moderate strength, gives chaotic starts that the model narrates or interprets;
  - noise that is too strong gives perseveration;
  - noise that is too weak is ignored.
- **The BOS intervention improves two measured outcomes:** fewer pre-reduction stops and fewer loop flags. It also concentrates slot-zero attention at bootstrap. Because it shifts positions, changes the protected fragment-token count, and advances reduction by one sample, these runs do not isolate the causal mechanism or establish that an attention sink is a prerequisite.
- **Memory policy measurably changes what comes next**, but these runs do not show which policy is "better".
- **Nothing here bears on experience or consciousness.** These are behavioral observations of text.

## Caveats

- Small n (2–8 settings per cell), a single sampling stream in most arms, and unblinded qualitative reading.
- Calibration comes from one fixture.
- Base model only (instruct comparison not run); T=1 only.

## Follow-up status

- Requested from Claude in #lab 20652: an 8B test with a more generous budget; separate plan and execution.
- Deferred: a gist-level residue test using source-embedding similarity vs a shuffled-fragment control, preregistered.
- Deferred: an instruct-model comparison and 16k-token runs unless included in the new follow-up plan.
- Deferred: putting `kv-dreaming/` under git.
