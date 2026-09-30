"""Unscaled Qwen rotate-half RoPE, adapted from memory's rotation helpers.

Positions may be fractional. Mixtures and inverse rotations use float32.
This deliberately does not implement scaled/dynamic RoPE.
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def rotate(x: torch.Tensor, positions, theta: float | None = None, *, inverse=False,
           inv_freq=None):
    work = x.float()
    p = torch.as_tensor(positions, dtype=torch.float32, device=x.device)
    if inv_freq is None:
        if theta is None:
            raise ValueError("provide model inv_freq or an unscaled theta")
        # Compatibility for standalone tests; runtime paths use model buffers.
        inv_freq = 1.0 / (theta ** (torch.arange(0, x.shape[-1], 2, device=x.device).float()
                                  / x.shape[-1]))
    inv_freq = inv_freq.to(device=x.device, dtype=torch.float32)
    if inv_freq.shape != (x.shape[-1] // 2,):
        raise ValueError("rotary frequency dimension mismatch")
    phase = p[:, None] * inv_freq[None, :]
    phase = torch.cat((phase, phase), dim=-1)
    sign = -1 if inverse else 1
    return work * phase.cos() + sign * rotate_half(work) * phase.sin()
