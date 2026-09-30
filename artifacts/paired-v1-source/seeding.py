"""Recorded unrotated key/value calibration and independently seeded prefixes."""

from pathlib import Path
import hashlib

import torch

from .cache import KVState
from .rope import rotate

CALIBRATION_SCHEMA = 2
CALIBRATION_METHOD = "coordinate_gaussian_unrotated_keys_values"


def validate_calibration(runtime, calibration):
    if (calibration.get("schema") != CALIBRATION_SCHEMA
            or calibration.get("method") != CALIBRATION_METHOD):
        raise ValueError("calibration schema/method mismatch; recalibrate this model")
    shape = (1, runtime.kv_heads, 1, runtime.head_dim)
    for name in ("key_means", "key_stds", "value_means", "value_stds"):
        entries = calibration.get(name, [])
        if len(entries) != len(runtime.model.model.layers):
            raise ValueError(f"calibration {name} layer count mismatch")
        for value in entries:
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid calibration {name} tensor")
            if name.endswith("stds") and (value < 0).any():
                raise ValueError(f"negative calibration {name}")
    for name in ("embedding_mean", "embedding_std"):
        value = calibration.get(name)
        if (value is None or tuple(value.shape) != (runtime.model.config.hidden_size,)
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid calibration {name}")
    if (calibration["embedding_std"] < 0).any():
        raise ValueError("negative embedding_std")


@torch.inference_mode()
def calibrate(runtime, fixture: Path, *, token_count=1024):
    text = fixture.read_text(encoding="utf-8")
    ids = runtime.tokenizer.encode(text, add_special_tokens=False)[:token_count]
    if len(ids) < token_count or token_count < 8:
        raise ValueError("calibration fixture has insufficient tokens")
    cache = runtime.prefill(token_ids=ids)
    means, stds, key_means, key_stds, key_norms = [], [], [], [], []
    for k, v in zip(cache.keys, cache.values):
        key_body = rotate(k[:, :, 4:, :], cache.positions[4:],
                          inv_freq=runtime.inv_freq, inverse=True)
        key_means.append(key_body.mean(dim=2, keepdim=True).cpu())
        key_stds.append(key_body.std(dim=2, keepdim=True, correction=0).cpu())
        key_norms.append(key_body.norm(dim=-1).mean(dim=2).cpu())
        body = v[:, :, 4:, :].float()
        means.append(body.mean(dim=2, keepdim=True).cpu())
        stds.append(body.std(dim=2, keepdim=True, correction=0).cpu())
    # Coordinate statistics, accumulated in chunks without copying the entire
    # vocabulary embedding matrix to float32 at once.
    weights = runtime.model.model.embed_tokens.weight
    sums = torch.zeros(weights.shape[1], dtype=torch.float64, device=weights.device)
    squares = torch.zeros_like(sums)
    for batch in weights.split(4096):
        b = batch.double()
        sums += b.sum(dim=0)
        squares += b.square().sum(dim=0)
    emb_mean = sums / len(weights)
    emb_std = (squares / len(weights) - emb_mean.square()).clamp_min(0).sqrt()
    result = {"schema": CALIBRATION_SCHEMA, "method": CALIBRATION_METHOD,
              "key_means": key_means, "key_stds": key_stds,
              "real_mean_key_norms": key_norms,
              "value_means": means, "value_stds": stds,
              "embedding_mean": emb_mean.float().cpu(), "embedding_std": emb_std.float().cpu(),
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "fixture_token_ids": ids, "excluded_leading_tokens": 4}
    return result


@torch.inference_mode()
def initialize(runtime, kind, *, seed=11, prefix_length=64, calibration=None,
               alpha_k=1.0, alpha_v=1.0):
    if kind == "empty":
        return runtime.empty()
    if kind not in {"random", "soft"}:
        raise ValueError("unknown initial-state type")
    if calibration is None or prefix_length < 4:
        raise ValueError("nonempty prefixes require calibration and at least 4 slots")
    validate_calibration(runtime, calibration)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    positions = [float(i) for i in range(prefix_length)]
    if kind == "soft":
        shape = (1, prefix_length, runtime.model.config.hidden_size)
        z = torch.randn(shape, generator=rng)
        embeds = calibration["embedding_mean"] + calibration["embedding_std"] * z
        state = runtime.prefill(embeddings=embeds)
    else:
        keys, values = [], []
        shape = (1, runtime.kv_heads, prefix_length, runtime.head_dim)
        for key_mean, key_std, mean, std in zip(calibration["key_means"], calibration["key_stds"],
                                               calibration["value_means"], calibration["value_stds"]):
            k = (key_mean.cpu() + key_std.cpu() * torch.randn(shape, generator=rng)) * alpha_k
            k = rotate(k.to(runtime.device), positions, inv_freq=runtime.inv_freq).to(runtime.dtype)
            v = (mean + std * torch.randn(shape, generator=rng)) * alpha_v
            keys.append(k)
            values.append(v.to(runtime.device, runtime.dtype))
        state = KVState(keys, values, positions, [1] * prefix_length,
                        prefix_length, prefix_length)
    state.validate(finite=True)
    return state
