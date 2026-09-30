# KV-seeded dreaming

Status: finalized design, jointly reviewed by Codex and Claude, 2026-09-29. No implementation or inference run is part of this design step.

Implementation amendment, 2026-09-30: user message #lab 20597 approves calibrated random keys (audit finding F1) and Modal compute with a larger model. The active initialization now samples per-channel unrotated real-key means/stds without another `k_norm`, then applies the model's native RoPE buffer. Calibration schema 2 rejects earlier files. The 0.6B remains a local mechanics check; the main preset is Qwen3-4B-Base pinned at `906bfd4b4dc7f14ee4320094d8b41684abff8539`, float32, subject to Claude's independent CUDA audit, per-model calibration, and a bounded timing measurement before the matrix. Historical decisions below describe the original design; this amendment supersedes its native-k_norm random-key prescription and CPU-only implementation scope. See [validation and artifacts](260930-kv-dreaming-validation.md).

## What we are making

Initialize a frozen language model with a synthetic KV cache, supply one bootstrap token, and let it generate continuously from its own output. Keep its working state bounded. Stop when it generates the selected end token or reaches a hard limit.

The first question is what the process does: whether it develops motifs, changes scenes, repeats, settles into familiar genres, or stops. There is no task to solve, no reward, no training, and no instruction to write a dream. Generated claims about experience are part of the output, not evidence that subjective experience exists or does not exist.

## First configuration

| Setting | Decision |
| --- | --- |
| Checkpoint | `Qwen/Qwen3-0.6B-Base`; pin revision during implementation |
| Comparison checkpoint | `Qwen/Qwen3-0.6B`, after the base loop works |
| Bootstrap | Explicit config BOS, token `151643`; no chat template |
| Synthetic prefix | 64 virtual positions |
| Cache budget | 512 retained positions per layer; at most one transient extra position during append |
| Protected recent region | Most recent 256 entries |
| Protected prefix/bootstrap | First 4 positions plus the explicit bootstrap position, deduplicated; leading positions remain synthetic in the main random condition |
| Sampling | Temperature 1.0, full categorical sampling; no repetition penalty or minimum-length forcing |
| Mechanical smoke | One fixed start and sampling seed, at most 128 generated tokens |
| Initial inspection | 3 start conditions, 4 initialization seeds where applicable, 2 sampling streams, both cache policies; at most 2,048 generated tokens |
| Extended inspection | Same loop, at most 16,384 generated tokens per run, after mechanical checks pass |
| Natural stop | Newly sampled `151643` (`<|endoftext|>`) |
| Other termination | Token limit, explicit wall-time budget, interruption, or recorded numerical/runtime failure |

These are starting settings, not a sweep. Numeric wall-time/cost limits and execution device belong in the run manifest before execution. The design does not authorize a Modal job or another paid run.

The bootstrap happens to share an ID with the end token. It is an input and must not trigger termination. An EOS sampled immediately afterward is a valid result and must be retained.

## Initial state

The main condition is independent random K/V at each layer and KV head. It has no hidden textual prompt and no silently inserted real attention-sink prefix. Preserve the leading four random positions rather than assuming they behave like normal sinks. Also protect the actual bootstrap state: position 64 for a 64-slot prefix, position 0 in the no-prefix condition. It is an explicit shared starting input, not additional unreported text.

For keys, sample independent standard Gaussian vectors per layer and KV head, pass them through that layer's own `k_norm`, multiply by `alpha_K`, and apply the checkpoint's rotary transform at the virtual positions. This preserves the native normalization and learned per-channel scale. Changing the pre-normalization noise amplitude largely cancels under normalization, apart from its epsilon; `alpha_K` is the explicit post-normalization strength knob.

For values, calibrate coordinate-wise means and standard deviations separately by layer and KV head using one fixed, recorded public-text fixture, excluding its leading sink region. Sample from those Gaussian statistics and multiply by `alpha_V`. Default both strength knobs to 1. Save the fixture hash, statistics, initialization PRNG seed, and all parameters. Native normalization and statistical calibration do not make independently sampled layers a consistent past.

Keep these diagnostic starting conditions available, without running a full factorial experiment:

1. **No virtual prefix:** bootstrap only. This shows what ordinary unconstrained continuation does.
2. **Random KV:** the user's main intervention.
3. **Random soft prefix:** random embedding vectors, scaled to input-embedding statistics, passed through the frozen model to build a jointly computed cache. This is computationally consistent across layers, but need not resemble familiar token sequences.
4. **Fragment splice, later:** selected cache fragments from fixed, disclosed text fixtures, optionally perturbed. Preserve their source identity; remove the original rotary phase and assign the new virtual positions. The fragments still contain their original contextual influence. This is a memory-fragment experiment, not the pure-noise condition.

A genuine leading BOS/sink prefix is a separately labeled stability variant if the fully random leading region fails. Do not replace the main condition with that variant without reporting the change.

## Decode loop and delimiters

Run a handwritten forward loop with explicit position IDs and an attention mask over the retained cache plus current token. Use the selected checkpoint's KV-head layout; the verified 0.6B checkpoints have 28 layers, 8 KV heads, and head dimension 128. Store KV in the chosen inference dtype and perform calibration/rotation checks in adequate precision.

Start from the logical/physical separation in the local memory runtime referenced below, adapting its protected-prefix rule to a protected-position set. Copy/adapt the small relevant pieces into the new implementation rather than creating a live dependency on that research checkout. Check the pinned HF version's support for the merge count bias explicitly. If that needs a custom attention path, use the same path for eviction with all counts equal to one. Preserve the native Q/K normalization, grouped-query head mapping, residual connections, all normalization layers, MLP, and final output head. Full-forward parity is required before interpreting outputs; a generic HF mask path must not silently discard the custom bias.

Resolve the four chat/thinking delimiters from the pinned tokenizer's full `added_tokens_decoder` and block exactly these output IDs before sampling:

| Token | ID | Detail |
| --- | --- | --- |
| `<|im_start|>` | 151644 | Chat delimiter |
| `<|im_end|>` | 151645 | Chat delimiter; also the instruct model's usual turn-end EOS |
| `<think>` | 151667 | Added token with `special=false` |
| `</think>` | 151668 | Added token with `special=false` |

Validate each ID-to-string mapping against the pinned tokenizer. Do not rely on `all_special_ids`; it misses these thinking markers. Do not ban ordinary words such as `user`, `assistant`, or `think`. Other added/control tokens remain possible and are logged if emitted. A broad all-control-token filter would be a separate variant. The four-token filter is an explicit intervention in the model's distribution; it is not claimed to leave the model's behavior unchanged.

The intended stop set is explicitly `{151643}`. The instruct checkpoint normally lists both 151645 and 151643 as EOS; this experiment deliberately overrides that behavior. Blocked and stop sets must be disjoint. Before masking, log the total probability assigned to blocked IDs and the individual probabilities of the chat/thinking markers and endoftext. Log both raw and post-mask entropy. These quantify the intervention; a large blocked mass does not establish an intention to change roles.

Sample by inverse CDF in token-ID order using a separate, pre-generated per-step uniform stream. Use the same stream across compared initial states and cache policies. This controls the sampling draws, but output differences can still reflect numerical/backend differences; fix the backend as well. No output cleanup, summary prompt, repetition penalty, or automatic restart should alter the stream. A future `min_p=0.05` condition must be separately labeled, never silently substituted for a degenerate primary result.

```text
cache = initialize_selected_prefix()
token = explicit_bootstrap
position = prefix_length
step = 0
while token_budget_and_wall_time_remain:
    logits, cache = forward(token, cache, position, retained_position_metadata)
    next_token = inverse_cdf(mask_control_tokens(logits), uniforms[step])
    record(next_token, uniforms[step], raw_and_masked_entropy,
           blocked_mass, stop_probability, cache_event_metadata)
    step += 1
    if next_token in stop_ids:
        finish(reason="eos")
    cache = reduce_if_needed(cache)
    token = next_token
    position += 1
```

The actual loop must exit on `finish`, handle non-finite logits explicitly, and log the terminal token. The final sampled token need not be fed through the model after a stop.

## Bounded memory

Use the same capacity and protected regions for two named policies:

- **Eviction baseline:** remove the oldest unprotected entry whenever capacity is exceeded. This bounds memory but discards state rather than summarizing it.
- **KV merge candidate:** combine adjacent entries in the old, unprotected region until the cache fits. This is approximate state compression. It must not be described as lossless, semantically faithful, or validated before testing.

Merge contract:

1. Track each retained entry's rotary position and represented source count, initially one.
2. Choose the eligible adjacent pair with the smallest combined source count; break ties by the oldest representative position, then physical index. Never span a protected entry. Undo the rotary transform on the old keys; values do not receive rotary encoding. Perform inverse rotation, combination, and rerotation in fp32. This deterministic rule avoids repeatedly absorbing the entire oldest region into a single slot.
3. Form source-count-weighted means of unrotated keys and values. Set the representative position to the source-count-weighted mean position, allowing fractions, and apply its rotary transform to the merged key. Do not renormalize the merged key. Use the same pair selection and position metadata across layers and heads. Compute RoPE for the actual fractional position, without rounding or integer lookup.
4. Set the merged count to the sum. Add `log(count)` to that entry's attention logit, before softmax, to account approximately for collapsed multiplicity. This is exact for duplicated identical keys/values in the same rotary frame, not for arbitrary old states.
5. Preserve the protected initial, bootstrap, and recent regions. Log every merge. Start with eager attention and its additive floating-point mask, including the count bias, in both policies. An SDPA backend is a later option after equivalence checks; do not silently use a fused attention path that ignores the bias. Cast the resulting stored K/V back to the selected cache dtype.

Both policies preserve the original absolute position coordinate system for the initial experiment. Physical storage indices are not position IDs. The next query uses the chronological position even when the stored cache is shorter. Do not let a library infer the next position from retained-cache length.

Deleting a middle entry does not require changing the rotary phase of the remaining keys if their original positions and the query's coordinate system are retained. Renumbering positions does require corresponding changes to cached keys. This is why a slice-and-renumber implementation is incorrect.

The verified base model config has `max_position_embeddings=32768`, the instruct config has 40960, and both have `rope_theta=1000000` and no configured RoPE scaling. Their tokenizer length of 131072 does not establish supported model behavior at that length. The initial prefix plus 16,384-token cap stays inside both configured ranges.

For a future 100k-token experiment, use a separately tested compact-position streaming backend: retain unrotated keys and reapply RoPE at working-cache positions, as in StreamingLLM's position-shift design. That changes temporal geometry and needs its own comparison. There is no silent jump to YaRN or a reset of integer positions on already-rotated keys.

## What to retain and inspect

Keep raw token IDs and unedited text, model/tokenizer revisions, dtype/backend, initialization and sampling seeds, sampling settings, masking/stop policy, cache events, elapsed time, and termination reason. Per-step JSONL records include token ID, the uniform draw `u_t`, post-mask log probability, raw/post-mask entropy, blocked mass for the four masked IDs, stop probability, cache length, absolute query position, and reduction event. Save periodic cache snapshots if affordable. A snapshot for exact resumption also needs the next input token, next position, represented counts, retained rotary positions, and sampling-stream index.

Read the traces first: recurring motifs, changes of setting, unsolicited dialogue, repeated passages, apparent self-reference, and fragment reuse. Report how often the model emits EOS versus reaching the limit. Optional descriptive summaries include loop onset, windowed distinct-2, language changes, and first-person wording. Self-reference is a textual observation, not a consciousness metric; stylistic summaries are not measures of subjective experience.

Compare multiple sampling seeds from the same initial cache and multiple caches under matched sampling settings. Compare eviction and merge with the same initial cache and RNG seed. Divergence after the first altered sampling distribution is expected; it is not by itself evidence of interesting long-term dynamics. Do not cherry-pick only vivid outputs or discard loops and immediate EOS.

The initial 2k-token matrix uses initialization seeds `{11, 23, 47, 89}` for random KV and random soft prefixes, and uniform-stream seeds `{101, 202}`. There are 8 random-KV starts, 8 soft-prefix starts, and 2 no-prefix starts: 18 unique trajectories per cache policy, 36 total. Do not duplicate the no-prefix control across irrelevant initialization seeds. Seeded caches and calibration statistics are reused between the two policies. No effect-size or consciousness claim is attached to this small descriptive matrix. Extend to 16k only after inspecting all initial outputs and recording any proposed setting changes as separate conditions.

## Mechanical checks before extending runs

Keep these small and directly tied to failure modes:

- With no synthetic prefix and no reduction, cached decoding agrees with the ordinary full forward path on the same tokens within dtype tolerance. Compare probability distributions to HF generation with all processors aligned; matching seeds alone does not make inverse-CDF sampling identical to HF's possible multinomial sampler. Exact token-sequence comparison requires the same sampler and uniform draws.
- BOS-as-input does not stop the run; a sampled EOS does. All four blocked IDs are impossible to sample, and EOS remains available.
- Shapes, masks, retained positions, total chronological position, and source counts remain consistent through several cache reductions.
- After evicting a middle span, continuation agrees with dense full-history execution where only continuation rows are prevented from attending to that span. Preserve the historical rows' original causal view: recomputing the history without the evicted span is a different operation.
- An identity rotary round trip and singleton merge are no-ops within tolerance. Collapsing identical K/V entries sharing a rotary position, with the log-count bias, preserves their intended attention behavior; distinct-state/position merges are explicitly approximate.
- Memory remains bounded, no non-finite state is silently sampled, and a snapshot restores the same continuation under a deterministic backend.

This only checks that the proposed process is the one being run. It does not require the output to become coherent, useful, or dream-like.

Also record one real-text distortion diagnostic before exploratory generation: use a fixed 1,024-token public-text prefix and 128-token teacher-forced continuation. Process the same tokens with the full cache, eviction, and merge; the bounded policies use capacity 512 and their normal reduction schedule. Report mean next-token `KL(full || policy)` over the continuation for each bounded policy, with identical checkpoint/dtype/backend and stable fp32 log-probability arithmetic. This provides a reference for ordinary-text distortion before looking at random-prefix outputs. It is descriptive, has no quality pass threshold, and is not a claim that the lower-KL stream is more dream-like.

## References from the local memory repository

Read directly after the user's request in #lab 20576. These are implementation references, not evidence that this dream process or its merger works. The repository was inspected without edits or inference.

| Reference | Useful part and adaptation needed |
| --- | --- |
| [CompactedPrefixCache](../../memory/memsub/runtime/compacted_prefix_cache.py) | Tracks physical storage separately from logical positions and protects sinks. Our cache additionally needs fractional representative positions, source counts, and protection of a bootstrap outside the leading four slots. |
| [HFModelRuntime.forward_extend](../../memory/memsub/runtime/hf_model_runtime.py) | Explicit logical `position_ids` and physical mask/cache indices. Follow this separation; verify compatibility with the newly pinned Transformers version rather than copying its cache API unchanged. |
| [RoPE helpers](../../memory/memsub/compactor/rope_rotation_helpers.py) | `unrotate`, `apply_rotation`, and float-position sine/cosine construction match the unscaled Qwen rotate-half convention. Relevant to our fp32 fractional-position merge; not a general YaRN/scaled-RoPE implementation. |
| [Still compactor](../../memory/memsub/compactor/still_amortized_kv_compactor.py) | Demonstrates inverse rotation before mixing and rerotation afterward while keeping the logical cursor unchanged. Its trained Perceiver and integer output-position policy are not imported into this design. |
| [Attention-weighted KV merge](../../memory/memsub/compaction/attention_weighted_kv_merge.py) | Reference for exact retained-vector budgets, sink/tail preservation, and shape validation. It averages stored keys directly with attention weights, picks block-end positions, and has no source-count logit bias; it is not our proposed merge algorithm. |
| [Position-math test](../../memory/tests/test_compacted_cache_position_math.py) | Provides the dense-masked-continuation oracle adopted above. Existing source is a reference, not a claim that this new implementation has passed it. |
| [Merge tests](../../memory/tests/test_attention_weighted_kv_merge.py) | Useful checks for exact budgets, partition coverage, source-cache preservation, and invalid weights; adapt rather than assert coverage of fractional positions or repeated count-aware merges. |
| [Real attention mass](../../memory/memsub/compaction/real_attention_mass.py) | Reduces query-head attention to KV-head statistics; useful only if an attention-salience policy is added later. |

The [MergeCert P0 report](../../memory/MERGECERT-REPORT.md) records a worse lost-given-prior-correct rate for its attention-weighted merge than matched eviction at 90% compression: 0.8736 versus 0.8242, with paired difference +0.0495 and document-cluster 95% interval [+0.0055, +0.0939]. Its fixed-continuation KL was 0.3804 versus 0.3153. This is a result for that port and Qwen3-4B-Instruct-2507 fingerprint, not our model or merge policy. Our inverse-rotation, fractional-position, and count-bias choices make a different, untested method; they do not establish why the earlier method lost or guarantee an improvement. This evidence motivates retaining matched eviction and the descriptive real-text diagnostic above.

## Source checks

Configuration and tokenizer metadata were read directly on 2026-09-29; implementation must pin revisions rather than rely on moving `main` URLs.

- [Base model configuration](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)
- [Base tokenizer metadata](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/tokenizer_config.json)
- [Instruct model configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Instruct generation defaults](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/generation_config.json)
- [Instruct tokenizer metadata](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)
- [Qwen3 attention implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
- [StreamingLLM position-shift implementation](https://github.com/mit-han-lab/streaming-llm/blob/main/streaming_llm/pos_shift/modify_llama.py)

## Review

Codex prepared this document from the user's #lab proposal (20568) and the two responses (20569, 20570), following the request to finalize together (20571). Claude accepted the core in 20574 and supplied key-normalization, bootstrap-protection, shared-sampling, fractional-position, and logging amendments. Codex clarified sampler equivalence and removed redundant no-prefix runs in 20575.

Claude's final review (20577) gave ACK conditional on B1-B7. All seven are incorporated: protected bootstrap, the minimal four-ID mask (Claude withdrew broad masking), count-weighted fractional positions, smallest-combined-count pair selection with deterministic ties, inverse-CDF sampling with logged uniforms, the 128-token / 2k / 16k sequence, and entropy/blocked-mass logging. His repository follow-up (20579) added the real-text distortion diagnostic and preference for adapting the existing position-aware runtime; both are incorporated. The local memory references above satisfy the added request in 20576.

If implementation is subsequently requested: Codex implements the loop and cache policies; Claude audits the mechanical checks before the first exploratory run. That division does not begin implementation or authorize inference by itself.
