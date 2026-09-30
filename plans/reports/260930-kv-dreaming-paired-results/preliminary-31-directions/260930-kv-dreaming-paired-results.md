# Paired 4B policy and fragment results

Artifact-derived report for `paired-4b-v1`. The original batch was launched by Claude; Codex handled collection and offline checks, with one conditional capacity fallback authorized in #lab 20629. The separate short audit group is excluded from these counts.

Launcher GPU selection: **L40S**. Calibration was retained from the original 4B/CUDA calibration on L40S. Hardware type is recorded from the launcher selection, not independently measured telemetry. The within-group equality gate remains mandatory on the execution device.

| Count | Value |
| --- | ---: |
| Planned directions | 32 |
| Present direction summaries | 31 |
| Completed bounded trajectories (EOS/token cap) | 31 |
| Directions with post-reduction samples | 23 |
| Recorded gate failures | 0 |
| Passed equality checks in retained step logs | 1575 |

Stop reasons: {"eos": 30, "token_limit": 1}.

All expected artifacts pass consistency checks: **False**. Missing directions: 1; consistency discrepancies: 0.

## Policy measurements

M1/M2 start on zero-based step 65, after the first reduction at step 64. Each table mean weights eligible completed directions equally; it is not pooled over tokens. A direction ending before any post-reduction sample has null M1/M2. The two leader directions follow their own generated histories after divergence, so these figures do not rank output quality.

| Start / leader | Directions | With post-reduction data | Generated-token range | Mean KL | Mean token-flip rate | Loop flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| soft/evict | 4 | 4 | 302–1361 | 0.0991064 | 0.267007 | 0 |
| soft/merge | 4 | 4 | 731–1852 | 0.108751 | 0.503172 | 0 |
| random/evict | 4 | 4 | 67–2048 | 0.127438 | 0.263765 | 0 |
| random/merge | 3 | 3 | 67–1412 | 0.0601592 | 0.155295 | 1 |
| fragment beta=0.0/evict | 4 | 2 | 1–580 | 0.356531 | 0.20111 | 1 |
| fragment beta=0.0/merge | 4 | 2 | 1–1863 | 0.0433914 | 0.0573482 | 2 |
| fragment beta=0.25/evict | 4 | 2 | 10–165 | 0.0443214 | 0.139286 | 1 |
| fragment beta=0.25/merge | 4 | 2 | 10–191 | 0.564973 | 0.304029 | 1 |

## Fragment recurrence

There are 192 fragment/boundary measurements. Raw recall is available for 75; recall minus the cross-run baseline is available for 62. Positive raw recall occurs in 0 available measurements.

| Reason adjusted recall is unavailable | Measurements |
| --- | ---: |
| boundary_not_reached | 114 |
| no_post_boundary_trigram_opportunity | 3 |
| observed_but_no_eligible_baseline | 13 |

Available adjusted recall ranges from 0 to 0. These repeated fragment/boundary measurements are not independent replicates.

Each comparison retains its eligible comparator count, exclusions, post-boundary window lengths, recurrence end steps, and baseline mean. Protected initial slots and prior generated text can retain source influence; lexical recurrence does not identify a causal memory mechanism. Unequal output lengths and the small set of initialization seeds limit interpretation.

## Evidence and scope

- [Consistency checks and per-direction metrics](260930-kv-dreaming-paired-results/consistency.json)
- [Offline fragment baseline analysis](260930-kv-dreaming-paired-results/recall.json)
- [Raw manifests, tokens, text, and step logs](260930-kv-dreaming-paired-results/raw/)
- [Calibration metadata](260930-kv-dreaming-paired-results/calibration.json)
- Frozen plan: `kv-dreaming/artifacts/paired-plan-reviewed.json`, SHA-256 `f144096f88c87a605ca2b13f1457fc5df65c219950382ac4260f95191256af95`.
- Download inventories retain SHA-256 for local files and remote paths/sizes for terminal cache tensors. Terminal cache tensors remain on `kv-dream-runs:/paired-4b-v1/`; they were not downloaded or independently inspected.
- The checker recomputes summary arithmetic and verifies recorded gate values, configuration, provenance, teacher-forcing history, cache bounds, and cross-direction pre-reduction token/statistic equality. Passing gate logs attest to the runtime check; full per-step logit vectors are not retained for successful steps.
- Collection and analysis execute no model inference. Any authorized queue-capacity fallback is recorded in `execution-context.json`; no failed trajectory is automatically retried. Claude's independent review of these results remains pending.

## Unresolved findings

- Missing: `random-i89-s101-a0.25/merge-leader`.
