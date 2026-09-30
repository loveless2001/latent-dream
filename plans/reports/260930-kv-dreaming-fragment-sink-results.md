# BOS fragment-start follow-up results

The frozen matched comparison returns **STOP_FRAGMENT_STARTS** for `paired-4b-sink-v1`.

All sixteen BOS directions completed. The comparison uses the sixteen retained no-sink fragment directions, matched on seed, beta, leader policy, model/backend/calibration, source text, fragment token IDs, and source offsets. Each cohort contains eight initial-cache groups; the two leader directions are not independent replicates. Claude launched the audited batch on L40S; Codex collected artifacts and ran offline checks. GPU type follows the launch record rather than independently measured telemetry.

| Preregistered outcome | No sink | BOS | Required change | Passed |
| --- | ---: | ---: | --- | --- |
| P1: No reduction reached | 8 / 16 | 0 / 16 | Strict decrease | True |
| P2: Low-diversity loop flag | 5 / 16 | 2 / 16 | Strict decrease | True |
| P3: Pre-policy fragment-trigram hit | 2 / 16 | 2 / 16 | Strict increase | False |

The agreed rule requires all three strict improvements. This complete experiment fails that rule, so stop this fragment-start line under the frozen design. The result does not reject every attention-sink hypothesis. Any narrower gains below remain descriptive and do not waive the stop rule.

P1 counts trajectories with no reduction event, including early EOS. P2 uses the existing 128-token window, unique-token fraction below 0.12, and stride 32. P3 requires an eligible selected-fragment trigram wholly inside samples zero through the first reduction step inclusive, or the entire output for earlier EOS. Outputs shorter than three tokens count as no hit and retain an opportunity flag.

P3 hits occur in both leader directions of seed 23, beta 0.25 without BOS, and both leader directions of seed 47, beta 0.25 with BOS. The matched aggregate therefore ties despite a change in which seed supplies the hits.

| Availability and validation | No sink | BOS |
| --- | ---: | ---: |
| Directions with post-reduction samples | 8 | 16 |
| Post-reduction samples | 2805 | 17084 |
| P3 windows shorter than three samples | 2 | 0 |

BOS stop reasons: {"eos": 12, "token_limit": 4}. All 1024 recorded pre-reduction equality checks pass with zero raw-logit difference. All 8 cross-direction prefix comparisons pass. Missing directions: 0; consistency discrepancies: 0; gate-failed directions: 0.

The first reduction occurs at zero-based step 63 with BOS, versus 64 without it. Policy-sensitive samples therefore begin at steps 64 and 65 respectively. Initial slots 0–3 and bootstrap position 65 remain protected in the BOS variant. The independent event replay checks the protected slots, recent tail, and fragment boundaries.

## Descriptive policy measurements

Each mean below weights directions with post-reduction data equally. The leader directions generate different histories after divergence; KL and hypothetical shared-uniform token flips do not rank output quality.

| Beta / leader | Directions with post-reduction data | Generated-token range | Mean KL | Mean token-flip rate | Loop flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| fragment beta=0.0/evict | 4 / 4 | 190–2048 | 0.119532 | 0.386351 | 0 |
| fragment beta=0.0/merge | 4 / 4 | 220–1845 | 0.0547152 | 0.294704 | 1 |
| fragment beta=0.25/evict | 4 / 4 | 889–2048 | 0.11478 | 0.490351 | 1 |
| fragment beta=0.25/merge | 4 / 4 | 131–2048 | 0.0686117 | 0.395238 | 0 |

## Post-boundary fragment recurrence (M4)

Of 192 fragment/boundary measurements, raw recall is available for 176, and recall minus the matched cross-seed baseline is available for 176. There are 9 positive raw-recall measurements, spanning 3 fragment/direction combinations in 2 directions. Their (seed, beta) groups are [(11, 0.25)]. Multiple boundary definitions reuse the same fragment trajectory, so the count is not a count of independent successes.

| Reason adjusted recall is unavailable | Measurements |
| --- | ---: |
| boundary_not_reached | 16 |

Available adjusted recall ranges from 0 to 0.0714286.

A separate standard-library replay checked all 34200 reduction events, all 192 fragment/boundary measurements, comparator eligibility, and baseline arithmetic with 0 discrepancies. Baselines match sink mode and prefix length. Protected fragment slots, earlier generated text, unequal output lengths, and the small set of seeds limit interpretation. These repeated boundary measurements are not independent replicates; lexical recurrence does not identify causal memory.

## Separate bootstrap attention diagnostic

Claude's separate diagnostic measures slot-zero attention at the bootstrap forward, averaged over layers, heads, and four seeds. The retained 28 logit-replica checks all have maximum absolute difference zero.

| Fragment beta | No-sink slot-zero mass | BOS slot-zero mass |
| --- | ---: | ---: |
| 0 | 0.0038 | 0.7716 |
| 0.25 | 0.0042 | 0.7480 |

This demonstrates concentrated slot-zero attention in this measured forward. It does not establish the absence of sinks at other positions or later steps, explain output quality, or change P1–P3. [Diagnostic data](claude-260930-0215-kv-dreaming-paired-results-audit/slot0-attention-4b.json), SHA-256 `664e2b7bc6b9ca1b4c9148c58b6a57e33841f0f63ec06d95500a9e9d53258da6`; [independent diagnostic script](claude-260930-0215-kv-dreaming-paired-results-audit/kv-dreaming-bootstrap-attention-mass-on-slot-zero-diagnostic.py).

## Scope and evidence

Adding BOS also shifts fragment/bootstrap positions, protects three rather than four fragment tokens, and advances the first reduction by one sample. This compound intervention cannot isolate an attention-sink mechanism. Claude's separate slot-zero attention diagnostic is descriptive and cannot change the primary criteria. [Claude's independent audit addendum](claude-260930-0215-kv-dreaming-paired-results-audit.md) confirms P1–P3, the 1,024 exact gates, the comparison/consistency/recall hashes, and the stop verdict.

- [Frozen preregistration](260930-kv-dreaming-fragment-sink-preregistration.md)
- [Matched primary comparison](260930-kv-dreaming-fragment-sink-results/comparison.json)
- [Consistency checks and per-direction metrics](260930-kv-dreaming-fragment-sink-results/consistency.json)
- [M4 analysis](260930-kv-dreaming-fragment-sink-results/recall.json)
- [Separate event/token replay](260930-kv-dreaming-fragment-sink-results/fragment-crosscheck.json) and [script](260930-kv-dreaming-fragment-sink-results/crosscheck_fragments.py)
- [Raw manifests, tokens, text, and step logs](260930-kv-dreaming-fragment-sink-results/raw/)
- [Frozen plan](260930-kv-dreaming-fragment-sink-results/plan.json), [freeze packet](260930-kv-dreaming-fragment-sink-results/freeze.json), and [calibration metadata](260930-kv-dreaming-fragment-sink-results/calibration.json)
- [Launcher index](260930-kv-dreaming-fragment-sink-results/launcher-index.json)

The checker validates recorded gates; full successful-step logits were not retained. Terminal cache tensors remain on `kv-dream-runs:/paired-4b-sink-v1/` and were not downloaded or independently inspected. Download inventories bind the local logs with SHA-256 and retain remote tensor paths/sizes. Collection and analysis ran no model inference.

| Artifact | SHA-256 |
| --- | --- |
| plan.json | `7c3d4e9b3ef519bb95a243ab74d7f1520b720f5101c7f7da04d481ae838968c3` |
| freeze.json | `b913870a8350f0d22d9efee938470331de1caca299b9277cf60eede3450fe33a` |
| comparison.json | `afaa660bc16ca9b24db7273592e1e344960777c30ae1f1ef4a54b1a468f51721` |
| consistency.json | `1972fcc4d53eac56ee6f22986134969c958624111e918e80015e6df5691409cd` |
| recall.json | `4aecb932d0fcaa3e60f49e08e597a97e0cbaa75efc68dd37502e6b8c8ce6da74` |
| fragment-crosscheck.json | `8959bfff90275f3bd7784b7ab11c9b236673033253961addce48c75ae1c1dec1` |
