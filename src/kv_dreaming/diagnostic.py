"""Fixed-token KL diagnostic, not a dream-quality gate."""

import time
import torch

from .cache import reduce_cache


@torch.inference_mode()
def continuation_kl(runtime, token_ids, *, prefix=1024, continuation=128,
                    budget=512, recent=256, max_seconds=900):
    if len(token_ids) < prefix + continuation:
        raise ValueError("fixture has insufficient diagnostic tokens")
    if prefix + continuation > runtime.model.config.max_position_embeddings:
        raise ValueError("diagnostic exceeds model context")
    ids = token_ids[:prefix + continuation]
    start = time.monotonic()
    # Keep only the continuation prediction rows plus the unused final row.
    # This avoids materializing full-vocabulary logits for the entire prefix.
    reference = runtime.model(torch.tensor([ids], device=runtime.device), use_cache=False,
                              logits_to_keep=continuation + 1).logits
    # The first retained row is at prefix-1 and predicts continuation token 0.
    reference = reference[0, :continuation].float().log_softmax(-1).cpu()
    result = {"prefix_tokens": prefix, "continuation_tokens": continuation,
              "budget": budget, "recent": recent, "direction": "KL(full || policy)", "policies": {}}
    for policy in ("evict", "merge"):
        state = runtime.empty()
        values, events = [], 0
        for i, token in enumerate(ids[:-1]):
            if time.monotonic() - start > max_seconds:
                raise TimeoutError("real-text diagnostic wall-time limit reached")
            logits = runtime.step(token, state)
            events += len(reduce_cache(state, policy=policy, budget=budget, recent=recent,
                                       theta=runtime.theta, inv_freq=runtime.inv_freq))
            if i >= prefix - 1:
                ref = reference[i - (prefix - 1)]
                candidate = logits.cpu().log_softmax(-1)
                values.append(float((ref.exp() * (ref - candidate)).sum()))
        result["policies"][policy] = {"mean_kl": sum(values) / len(values), "per_token_kl": values,
                                      "reduction_events": events, "cache_length": state.length}
    result["elapsed_seconds"] = time.monotonic() - start
    return result
