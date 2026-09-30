# Audit: paired-4b-v1 results (claude, 2026-09-30)

Independent check of codex's `260930-kv-dreaming-paired-results.md`, run from the raw step logs with no production imports. Script: scratchpad `independent-paired-audit.py`. Hashes match codex's post: consistency `6b1ae865…`, recall `d5366cc5…`.

## Verdict: report CONFIRMED

- **Gate (CUDA, including the reduction path):** 1,640 gate records, all `passed` with `max_abs_diff == 0.0`.
  - Gates are present exactly on steps ≤ first reduction and absent afterwards.
  - Pre-reduction KL is exactly 0 with no flips.
- **Teacher forcing:** follower input equals [BOS] + leader tokens in all 32 directions.
- **Cross-direction:** the evict-leader and merge-leader token prefixes are identical through the first reduction in all 16 groups.
- **M1/M2:** per-direction means and the 8-cell table match the report to 6 decimals. Flip field = (leader token ≠ follower token under the same u).
- **Issues found:** 0.

## Interpretation additions

1. **Policy matters to prediction.** After reduction, evict and merge memories give materially different next-token distributions: mean KL 0.04–0.56 nats, shared-u flip rate 6–50% across cells. There is no consistent direction, n=2–4 per cell, and some cells rest on 2-sample directions.
2. **The M4 zero is uninformative about memory policy.** The fragment start itself mostly failed:
   - immediate EOS (1 token), `slslsl…`, `","","",…`, `rectrectymym…`, `from it from it…`, `Abstract ×9`;
   - 8 of 16 fragment directions never reached a reduction.
   - Only i23-β0.25 reproduced fragment text at all ("model outputs can still differ", which comes from the `sampling` source fixture), and it ended at 19 tokens.
   
   So "no fragment 3-gram recurs after its slots are reduced" mostly reflects residue that never got expressed in the first place. It is not evidence that evict or merge erases it.
3. **Hypothesis for the fragment failure (untested): no attention sink.**
   - Fragments are cut from mid-text, so the spliced cache has no real sink state: slots 0–3 are fragment 0's mid-document tokens.
   - The soft start is produced by an actual prefill, so it has a natural position-0 sink, and it behaved well (0/16 loops).
   - Random KV also lacks a real sink, and at α=1 it looped.
   - This is consistent with the "real sink prefix" stability variant named in the design spec but never run.

## Proposed follow-up (needs user go)

- **Fragment + real sink:** prepend a genuine position-0 state (e.g. the bootstrap/BOS prefilled at virtual position 0), then the 4 fragments at 1..64, keeping the other settings.
- Same 4 seeds × β{0, 0.25} × both leaders, about 1 GPU-hour.
- **Prediction if the sink hypothesis holds:** early-EOS/loop rate drops, and fragment 3-grams appear before the boundary. Only then does M4 become a real policy test.

## Unresolved

- Is the sink-variant rerun worth doing, or should fragment starts be dropped?

## Addendum: fragment + BOS sink (paired-4b-sink-v1), audit of codex's packet

**Independent recompute** (script `independent-fragment-sink-p1-p3-recompute.py`, no production imports), matched to the frozen no-sink baseline:

| | no sink | BOS sink | preregistered need |
|---|---|---|---|
| P1: directions that never reach a reduction | 8/16 | **0/16** | fewer ✓ |
| P2: loop flag (128-window unique <0.12) | 5/16 | **2/16** | fewer ✓ |
| P3: eligible fragment 3-gram through the first-reduction step | 2/16 | **2/16** | more ✗ |

- Exact gates: 1,024/1,024. Hashes match codex's post (comparison `afaa660b…`, consistency `1972fcc4…`, recall `4aecb932…`).
- **Preregistered verdict: STOP_FRAGMENT_STARTS. Confirmed.**

**Mechanism** (slot-0 diagnostic, bootstrap step only): the BOS slot draws ~0.75–0.77 of attention, versus ~0.004 without it.

**Unpreregistered observation (anecdotal, not a verdict):**
- With the sink, fragment starts now give long, coherent documents: a film review, a Python game class, a made-up "Project Gutenberg Etext of *Finch*, by Charles Dickens", and a biography of "biologist and explorer Robert Boyle".
- Several echo the *genre or gist* of the residue (Gutenberg framing, a Darwin-like naturalist theme, the sampling fixture's "model outputs across different processes") without matching 3-grams.
- The lexical P3 cannot see this. A gist-level test (e.g. source-embedding similarity against a shuffled-fragment control) would need its own preregistration. It is not a rescue of this result.
