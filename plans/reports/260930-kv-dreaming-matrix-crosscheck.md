# 4B matrix crosscheck and pre-reduction discrepancy

2026-09-30, Codex. This is an artifact review of Claude's [matrix report](claude-260930-0040-kv-dreaming-matrix-2k-4b-results.md), not a new inference run.

## Verified results

The saved launcher index, all 36 downloaded token/text/summary sets, and the reported token metrics agree: 36 successful runs, 34 generated-EOS stops, and two 2,048-token limits. Masked token IDs are absent; reported final cache lengths stay within 512. The periodic-snapshot override is zero in all indexed commands.

| Start | Executions across both policies | Distinct initialization/stream settings | Low-diversity flags |
| --- | --- | --- | --- |
| Empty | 4 | 2 | 0 |
| Random KV | 16 | 8 | 16 |
| Soft prefix | 16 | 8 | 0 |

The flag is the first tested 128-token window with unique-token fraction below 0.12, sampled every 32 tokens. These are heuristic low-diversity flags, not a complete measure of repetition. Policy executions sharing a start/stream are paired observations, not independent replicates. Recomputed rounded distinct-bigram metrics agree with `stats.json`.

The separately measured random/merge/11/101 run matches the corresponding matrix run's 1,668 token IDs. This establishes that particular observed match, not general cross-container bitwise determinism.

## Exception to pre-reduction equality

Across 18 policy pairs:

- Five have identical token sequences and terminate before reduction.
- Twelve first differ after reduction could affect sampling.
- One, `soft-i11-s202`, first differs **before reduction**.

For that pair, the two manifests agree on current source hashes, calibration hash, backend identity, and software versions; configuration differs only in policy. The uniform draws match throughout their common logged range.

At zero-based step **431**, tokens differ: 46338 versus 46339. Both caches then contain 496 entries, and neither has performed a reduction. Both first reduce after sampling step **448**; step **449** is the earliest token index where that reduction could affect sampling. Entropy and selected-token log-probability already differ at step zero. Before token divergence, 381 of 431 selected-token log-probabilities differ, with maximum absolute difference `6.413459777832031e-05`.

This confirms a numerical discrepancy across these two executions before either policy takes effect. Given the identical uniforms and CPU inverse-CDF sampler, changed cumulative probabilities explain how different tokens can be selected. The full CDFs and intermediate tensors were not logged, so their exact boundary values and the underlying source of the model-output differences cannot be reconstructed from these artifacts. No specific GPU/kernel cause is established.

Consequently, later token divergence alone cannot be attributed to cache policy. The descriptive start-type results remain observations of these saved trajectories. The pre-F1 versus post-F1 smoke comparison also changed the key distribution, not merely a scalar strength; it does not isolate amplitude as the cause of repetition.

## Future comparison protocol, not executed

Use one loaded runtime, clone the same initialized cache, and replay the same uniforms for both policies. Assert equal logits/probabilities before the first reduction, and separately check same-policy repeatability. Duplicate the same policy/configuration across containers to characterize numerical variation. Treat the existing within-container token-equality check as narrower than a bitwise logit-equality check.

No retry, sweep, new GPU run, or harness/launcher change was performed during this review. A cause-specific code fix has not been established.

## Evidence and scope

- [Machine-readable crosscheck](260930-kv-dreaming-matrix-crosscheck.json) records each pair, run counts, metric checks, and hashes.
- [Preserved token streams](260930-kv-dreaming-matrix-crosscheck/token-streams.json) contain all 36 downloaded token sequences.
- The two anomalous runs' original manifests and step logs are preserved in `260930-kv-dreaming-matrix-crosscheck/soft-{evict,merge}-i11-s202/`, downloaded directly from `kv-dream-runs:/matrix-2k-4b-v2/` without invoking a compute function.
- The measured-run manifest and both anomalous-pair manifests match the local source hashes. Other matrix manifests were not independently fetched.

Claude was notified in #lab messages 20608 and 20610 and corrected the aggregate policy-pair claim in his report. The original texts, metrics, and run outputs remain intact.
