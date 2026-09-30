"""Full-vocabulary T=1 inverse-CDF sampling with explicit uniforms."""

import math
import torch


def uniforms(seed: int, length: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(length, generator=generator, dtype=torch.float64).tolist()


def entropy(log_probabilities):
    p = log_probabilities.exp()
    finite = torch.isfinite(log_probabilities)
    return float(-(p[finite] * log_probabilities[finite]).sum())


def sample(logits, u: float, blocked_ids, stop_id: int):
    # One arithmetic path for identical logits; model outputs can still differ
    # across CPU/CUDA even when both model forwards are deterministic.
    logits = logits.detach().to(device="cpu", dtype=torch.float32)
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise FloatingPointError("sampling requires a finite logit vector")
    if not 0 <= u < 1 or not math.isfinite(u):
        raise ValueError("uniform draw must be in [0,1)")
    blocked_ids = sorted(set(blocked_ids))
    if stop_id in blocked_ids:
        raise ValueError("stop token cannot be blocked")
    if any(i < 0 or i >= len(logits) for i in blocked_ids + [stop_id]):
        raise ValueError("token ID outside vocabulary")
    raw = logits.float().log_softmax(dim=-1)
    masked_logits = logits.float().clone()
    masked_logits[blocked_ids] = -torch.inf
    masked = masked_logits.log_softmax(dim=-1)
    probabilities = masked.double().exp()
    probabilities /= probabilities.sum()
    cdf = probabilities.cumsum(dim=-1)
    cdf[-1] = 1.0
    token = int(torch.searchsorted(cdf, torch.tensor(u, device=cdf.device, dtype=cdf.dtype), right=True))
    return token, {"uniform": u, "log_probability": float(masked[token]),
                   "raw_entropy": entropy(raw), "masked_entropy": entropy(masked),
                   "blocked_mass": float(raw[blocked_ids].exp().sum()),
                   "blocked_probabilities": {str(i): float(raw[i].exp()) for i in blocked_ids},
                   "raw_stop_probability": float(raw[stop_id].exp()),
                   "stop_probability": float(masked[stop_id].exp())}
