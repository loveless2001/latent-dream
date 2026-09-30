"""Explicit logical coordinates, source counts, and bounded physical storage."""

from dataclasses import dataclass, field
import math

import torch

from .rope import rotate


@dataclass
class KVState:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    positions: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    next_position: int = 0
    bootstrap_position: int = 0

    @property
    def length(self):
        return len(self.positions)

    def clone(self):
        return KVState([k.clone() for k in self.keys], [v.clone() for v in self.values],
                       self.positions.copy(), self.counts.copy(), self.next_position,
                       self.bootstrap_position)

    def validate(self, *, finite=False):
        n = self.length
        if len(self.counts) != n or len(self.keys) != len(self.values):
            raise ValueError("cache metadata length mismatch")
        if any(c < 1 or not isinstance(c, int) for c in self.counts):
            raise ValueError("source counts must be positive integers")
        if any(not math.isfinite(p) or p < 0 or p >= self.next_position for p in self.positions):
            raise ValueError("invalid logical position")
        if any(a > b for a, b in zip(self.positions, self.positions[1:])):
            raise ValueError("cache positions are not ordered")
        for k, v in zip(self.keys, self.values):
            if k.shape != v.shape or k.ndim != 4 or k.shape[0] != 1 or k.shape[2] != n:
                raise ValueError("invalid layer KV shape")
            if finite and (not torch.isfinite(k).all() or not torch.isfinite(v).all()):
                raise FloatingPointError("non-finite KV state")

    def protected_indices(self, recent: int, initial: int = 4):
        protected = set(range(max(0, self.length - recent), self.length))
        protected.update(i for i, p in enumerate(self.positions)
                         if p < initial or p == self.bootstrap_position)
        return protected

    def select(self, indices):
        """Physical selection leaves logical positions and cursor unchanged."""
        self.keys = [k[:, :, indices, :].clone() for k in self.keys]
        self.values = [v[:, :, indices, :].clone() for v in self.values]
        self.positions = [self.positions[i] for i in indices]
        self.counts = [self.counts[i] for i in indices]

    def payload(self):
        return {"keys": [x.cpu() for x in self.keys], "values": [x.cpu() for x in self.values],
                "positions": self.positions, "counts": self.counts,
                "next_position": self.next_position, "bootstrap_position": self.bootstrap_position}

    @classmethod
    def from_payload(cls, payload, device="cpu"):
        result = cls(**{**payload, "keys": [x.to(device) for x in payload["keys"]],
                        "values": [x.to(device) for x in payload["values"]]})
        result.validate(finite=True)
        return result


def merge_pair(state: KVState, index: int, theta: float, *, inv_freq=None):
    """Approximate one adjacent pair; no key renormalization."""
    if index < 0 or index + 1 >= state.length:
        raise ValueError("merge pair is outside the cache")
    i, j = index, index + 1
    ca, cb = state.counts[i:j + 1]
    total = ca + cb
    old_positions = state.positions[i:j + 1]
    representative = (ca * old_positions[0] + cb * old_positions[1]) / total
    norm_log = []
    for layer, (k, v) in enumerate(zip(state.keys, state.values)):
        unrotated = rotate(k[:, :, i:j + 1], old_positions, theta, inverse=True,
                           inv_freq=inv_freq)
        merged_k = (ca * unrotated[:, :, :1] + cb * unrotated[:, :, 1:]) / total
        merged_v = (ca * v[:, :, i:i + 1].float() + cb * v[:, :, j:j + 1].float()) / total
        input_norms = unrotated.norm(dim=-1)[0]
        merged_norms = merged_k.norm(dim=-1)[0, :, 0]
        weighted_norms = (ca * input_norms[:, 0] + cb * input_norms[:, 1]) / total
        ratios = merged_norms / weighted_norms.clamp_min(torch.finfo(torch.float32).tiny)
        def summary(values):
            return {"mean": float(values.mean()), "min": float(values.min()),
                    "max": float(values.max())}
        norm_log.append({"layer": layer, "input_key_norms": summary(input_norms),
                         "merged_key_norms": summary(merged_norms),
                         "merged_to_weighted_input_ratio": summary(ratios)})
        kr = rotate(merged_k, [representative], theta, inv_freq=inv_freq).to(k.dtype)
        state.keys[layer] = torch.cat((k[:, :, :i], kr, k[:, :, j + 1:]), dim=2)
        state.values[layer] = torch.cat((v[:, :, :i], merged_v.to(v.dtype), v[:, :, j + 1:]), dim=2)
    state.positions[i:j + 1] = [representative]
    state.counts[i:j + 1] = [total]
    return {"kind": "merge", "index": i, "input_positions": old_positions,
            "input_counts": [ca, cb], "position": representative, "count": total,
            "key_norms": norm_log}


def reduce_cache(state: KVState, *, policy: str, budget: int, recent: int, theta: float,
                 inv_freq=None):
    if policy not in {"evict", "merge"}:
        raise ValueError("policy must be evict or merge")
    if budget < recent + 7 or recent < 1:
        raise ValueError("budget must allow recent tail, initial slots, bootstrap, and a merge pair")
    events = []
    while state.length > budget:
        protected = state.protected_indices(recent)
        if policy == "evict":
            eligible = [i for i in range(state.length) if i not in protected]
            if not eligible:
                raise ValueError("no evictable cache entry")
            i = min(eligible, key=lambda i: (state.positions[i], i))
            events.append({"kind": "evict", "index": i, "position": state.positions[i],
                           "count": state.counts[i]})
            state.select([j for j in range(state.length) if j != i])
        else:
            pairs = [i for i in range(state.length - 1)
                     if i not in protected and i + 1 not in protected]
            if not pairs:
                raise ValueError("no adjacent unprotected cache pair")
            i = min(pairs, key=lambda i: (state.counts[i] + state.counts[i + 1],
                                          state.positions[i], i))
            events.append(merge_pair(state, i, theta, inv_freq=inv_freq))
    state.validate()
    return events
