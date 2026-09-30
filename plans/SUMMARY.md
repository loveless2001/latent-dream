# KV-dreaming experiment so far

Updated 2026-10-01 from the #lab thread, retained launch indexes, and reports. The latest completed work is the 8B noise sweep. The [report index](README.md) links every phase; [summary-verification.json](summary-verification.json) records the locally checked counts and their source hashes.

The experiment implements the user's idea: a frozen base model starts with artificial KV memory, receives one bootstrap token, and then feeds its sampled output back into inference until it emits EOS or reaches a hard limit. It can produce varied, multilingual text that sometimes goes on to interpret its own earlier output. **Random embeddings passed through the model (the soft start) gave the most consistently varied, non-looping samples across the tested 4B and 8B configurations. Direct Gaussian KV injection was much more fragile.** “Dream-like” describes an unblinded reading of the text; the experiment does not measure experience or consciousness.

## What was built

- An explicit Qwen3 forward path with calibrated per-layer/head/channel KV initialization, deterministic inverse-CDF draws, raw text/token/step logs, and bounded cache handling. The local 0.6B checkpoint was used for mechanics and validation, with 4B and 8B experiments on Modal L40S in fp32.
- Empty, random-KV, soft-prefix, and real-text-fragment starts. The fragment condition imports external source material; the other nonempty starts use synthetic state. All conditions receive a bootstrap token.
- Eviction and approximate KV merging. Merging undoes RoPE, averages keys/values with source counts, uses a weighted fractional position, rerotates, and supplies a log-count attention bias. It is not lossless compression.
- Temperature 1, no repetition penalty, and masking of `<|im_start|>`, `<|im_end|>`, `<think>`, and `</think>`. Only newly sampled `<|endoftext|>` ends a run. Ordinary textual `Human:`/`Assistant:` turns can still appear.

The first agent proposals concerned verified rehearsal and memory/weight consolidation. The user redirected the project to this KV-seeded generation mechanism. Those earlier proposals are archived and were not implemented; no weights were trained in this campaign.

## Completed experiments

There are **128 bounded trajectories across six primary batches: 109 EOS stops and 19 token-cap stops**, with no reported runtime errors in those batches. This excludes local mechanics, smoke runs, timing checks, and separate audit groups. Executions sharing starts or leader directions are not independent replicates; reused survey points are counted once.

| Batch | Trajectories | Cache / recent | Token cap | EOS / cap | Main result |
| --- | ---: | --- | ---: | --- | --- |
| 4B initial matrix | 36 | 512 / 256 | 2,048 | 34 / 2 | Full-strength random KV looped in 16/16 executions; soft prefixes in 0/16; empty in 0/4 |
| 4B random-KV sweep | 12 new | 512 / 256 | 2,048 | 11 / 1 | Loop flags: alpha .1 = 0/4, .25 = 1/4, .5 = 0/4 |
| 4B paired policy/fragment test | 32 directions | 128 / 64 | 2,048 | 31 / 1 | 24 directions reached policy-comparison samples; all 1,640 pre-reduction equality checks passed |
| 4B fragment + BOS follow-up | 16 directions | 128 / 64 | 2,048 | 12 / 4 | Early stops and loops improved; the required lexical-hit count did not, giving STOP_FRAGMENT_STARTS |
| 8B generous survey | 20 | 2,048 / 1,024 | 8,192 | 16 / 4 | Soft remained non-looping; most full-strength random-KV runs stayed stuck to the cap |
| 8B random-KV sweep | 12 new | 2,048 / 1,024 | 8,192 | 5 / 7 | Loop flags in most seeds at every new alpha tested: .05, .1, .15 |

Sources: [4B matrix/sweep](reports/claude-260930-0040-kv-dreaming-matrix-2k-4b-results.md), [paired results](reports/260930-kv-dreaming-paired-results.md), [BOS results](reports/260930-kv-dreaming-fragment-sink-results.md), [8B survey/sweep](reports/claude-260930-1100-kv-dreaming-8b-generous-survey.md), and [retained launch indexes](../modal/runs-index/).

## What the starts do

**Soft prefixes:** random embeddings are passed through the model to create mutually consistent layer states. In the first 4B matrix, none of the 16 soft executions triggered the loop heuristic; neither did the four 8B soft runs. Samples often start with mixed-script fragments, then turn into explanations of those fragments, invented terminology, code, or questions and answers. This observed consistency makes soft starts the strongest practical candidate from the tested set, without establishing reliability across other seeds, temperatures, or models.

**Direct random KV:** on 4B, full calibrated strength gave low-diversity loops in all 16 initial executions. Reducing the scale to .1 usually produced ordinary document-like continuations, while .25–.5 produced varied samples and occasional self-interpretation. These are four seeds per sweep level, not an established optimum. The same pattern did not reproduce cleanly on 8B:

| 8B random-KV alpha | Loop flags | Token-cap stops |
| --- | --- | --- |
| .05 | 4/4 | 1/4 |
| .1 | 3/4 | 3/4 |
| .15 | 3/4 | 3/4 |
| .25, reused survey runs | 2/4 | 1/4 |
| 1, reused survey runs | 3/4 | 3/4 |

Some exceptions generated ordinary content or a Chinese poem. The tested 8B levels did not establish a reliably low-loop region. A proposed explanation involving attention dilution by near-zero KV entries remains untested. The comparison also changes cache size, token cap, and checkpoint details, so it does not isolate model size as the cause.

**Empty starts:** bootstrap-only generation tended toward ordinary code/documents on 4B. The 8B survey had one loop flag among four empty runs, so free-running generation can repeat even without synthetic KV noise.

## Memory policy and reproducibility

The first matrix had one eviction/merge pair whose tokens diverged before a cache reduction. The retained logs show a small numerical discrepancy despite matching source/backend metadata and uniform draws; they do not identify its hardware cause. The [crosscheck](reports/260930-kv-dreaming-matrix-crosscheck.md) therefore ruled out treating those separate-container continuations as a clean causal policy comparison.

The replacement experiment clones one initial state in one loaded runtime. One policy generates freely while the other receives the leader's token history; both leader directions are run. **All 2,664 recorded pre-reduction logit-equality checks passed across the two paired batches** (1,640 + 1,024). After reduction, eviction and merge give different next-token distributions and shared-draw outcomes. The original paired condition means range around .04–.56 nats KL and 6–50% token flips. These descriptive differences do not establish which policy makes better text or retains meaning better. A separate local 0.6B real-text fixture had lower mean distortion for eviction (.0338894 versus .0541150 KL), limited to that fixture.

The 8B survey used eviction only and does not compare eviction with merging. Its retained launcher summaries show **6/20 runs reached at least one reduction**: one empty, one fragment, and four random runs. This corrects the historical report's 5/20. Most survey runs ended before exercising bounded-cache behavior; the larger allowance did not establish better output quality. The [verification record](summary-verification.json) identifies the six runs.

## Real-text fragments and BOS

Four source fixtures supplied spliced KV fragments, with beta 0 or .25 noise. Without BOS, 8/16 fragment directions stopped before a reduction, and 5/16 had loop flags. Their available post-boundary lexical-recall measurements were all zero.

Adding a genuine BOS cache slot improved two preregistered outcomes, but the joint advance rule required all three strict improvements:

| Required outcome | No BOS | BOS | Result |
| --- | --- | --- | --- |
| P1: fewer directions failing to reach a reduction | 8/16 | 0/16 | Passed |
| P2: fewer loop flags | 5/16 | 2/16 | Passed |
| P3: more directions with an eligible source trigram through the first reduction | 2/16 | 2/16 | Failed |

**The audited verdict remains STOP_FRAGMENT_STARTS for the frozen 4B design.** P3 failed to increase; it did not show that verbatim recurrence never happens. Later M4 measurements include nine positive boundary rows from three fragment/direction combinations in two directions, all seed 11 / beta .25. They are not nine independent successes.

The separate attention diagnostic shows that BOS concentrates slot-zero attention at the bootstrap forward: about .004 without BOS versus .75–.77 with BOS. It does not measure every later position, establish that an attention sink is necessary, or isolate the mechanism behind the behavioral gains. Adding BOS also shifts positions, changes protection of fragment tokens, and advances reduction by one sample.

Some BOS outputs resemble the source's theme or genre, including invented Gutenberg/naturalist material. These are unblinded anecdotes; gist-level source attribution was not tested. The later 8B fragment survey was a separate descriptive user-requested arm, not a restart or rescue of the failed 4B criterion. Its outputs were often repetitive self-questioning, with one loop flag among four runs.

## Current boundary

The harness, initial surveys, noise sweeps, and paired/BOS analyses are complete. The 4B stop rule stands, and the requested 8B survey and sweep are complete. A gist-level residue test, an instruct-model comparison, and a 16k-token extension remain unstarted. The project is still not initialized as a git repository. This consolidation started no GPU work.

The evidence supports an internally seeded generation process, sensitivity to initialization, and measurable effects of cache policy. It does not establish learning, useful consolidation, persistent capability improvement, human-like dreaming, or subjective experience. Small samples, mostly one sampling stream, one calibration fixture, unblinded text reading, and heuristic loop flags limit the interpretation. Full terminal cache tensors remain on Modal and were not independently inspected locally.
