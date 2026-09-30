"""Hash-checked, separately contextualized source fragments at virtual positions."""

import hashlib
import json
import math
from pathlib import Path

import torch

from .cache import KVState
from .rope import rotate
from .seeding import validate_calibration


def trigrams(tokens):
    return set(zip(tokens, tokens[1:], tokens[2:]))


def load_sources(runtime, manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1 or len(manifest.get("sources", [])) != 4:
        raise ValueError("fragment manifest must disclose exactly four sources")
    if manifest.get("fragment_length") != 16:
        raise ValueError("fragment length must be 16")
    cap = manifest.get("source_token_cap")
    if not isinstance(cap, int) or not 16 <= cap < runtime.model.config.max_position_embeddings:
        raise ValueError("invalid source token cap")
    sources = []
    for row in manifest["sources"]:
        source = path.parent / row["file"]
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"fragment source hash mismatch: {row['id']}")
        full_ids = runtime.tokenizer.encode(raw.decode("utf-8"), add_special_tokens=False)
        ids = full_ids[:cap]
        if len(ids) < 16:
            raise ValueError(f"fragment source too short: {row['id']}")
        sources.append({**row, "token_ids": ids, "full_token_ids": full_ids})
    if len({s["id"] for s in sources}) != 4 or len({s["sha256"] for s in sources}) != 4:
        raise ValueError("fragment sources must have distinct IDs and content")
    if len({s["genre"] for s in sources}) < 3:
        raise ValueError("fragment sources must span at least three genres")
    return sources, hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def initialize_fragment(runtime, manifest_path, *, seed=11, beta=0.0, calibration=None):
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("fragment noise beta must be finite and nonnegative")
    if beta:
        if calibration is None:
            raise ValueError("fragment noise requires model-specific calibration")
        validate_calibration(runtime, calibration)
    sources, manifest_hash = load_sources(runtime, manifest_path)
    excluded = set.intersection(*(trigrams(s["full_token_ids"]) for s in sources))
    # Offset selection is separate from noise draws; beta changes neither the
    # chosen source fragments nor the underlying standard-normal draws.
    rng = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.Generator(device="cpu").manual_seed(seed ^ 0x4B564452)
    order = torch.randperm(4, generator=rng).tolist()
    selections = [(i, int(torch.randint(len(sources[i]["token_ids"]) - 15,
                                        (1,), generator=rng))) for i in order]
    keys = [[] for _ in runtime.model.model.layers]
    values = [[] for _ in runtime.model.model.layers]
    fragments = []
    for fragment_id, (source_index, offset) in enumerate(selections):
        source = sources[source_index]
        cache = runtime.prefill(token_ids=source["token_ids"])
        start = fragment_id * 16
        virtual = list(range(start, start + 16))
        original = cache.positions[offset:offset + 16]
        selected_ids = source["token_ids"][offset:offset + 16]
        eligible = trigrams(selected_ids) - excluded
        fragments.append({"id": fragment_id, "source_id": source["id"],
                          "source_sha256": source["sha256"], "source_offset": offset,
                          "original_positions": original, "virtual_positions": virtual,
                          "token_ids": selected_ids, "eligible_trigrams": [list(t) for t in sorted(eligible)],
                          "excluded_trigram_count": len(trigrams(selected_ids) & excluded),
                          "protected_positions": [p for p in virtual if p < 4]})
        for layer, (k, v) in enumerate(zip(cache.keys, cache.values)):
            content = rotate(k[:, :, offset:offset + 16], original,
                             inv_freq=runtime.inv_freq, inverse=True)
            value = v[:, :, offset:offset + 16].float().clone()
            if beta:
                for tensor, kind in ((content, "key"), (value, "value")):
                    mean = calibration[kind + "_means"][layer].cpu()
                    std = calibration[kind + "_stds"][layer].cpu()
                    draw = mean + std * torch.randn(tensor.shape, generator=noise)
                    tensor.add_(draw.to(runtime.device), alpha=beta)
            keys[layer].append(rotate(content, virtual, inv_freq=runtime.inv_freq).to(runtime.dtype))
            values[layer].append(value.to(runtime.dtype))
        del cache
    state = KVState([torch.cat(k, 2) for k in keys], [torch.cat(v, 2) for v in values],
                    list(map(float, range(64))), [1] * 64, 64, 64)
    state.validate(finite=True)
    metadata = {"schema": 1, "kind": "fragment", "manifest_sha256": manifest_hash,
                "initialization_seed": seed, "beta": beta,
                "noise": "beta * N(calibrated_mean, calibrated_coordinate_std), additive in unrotated space",
                "source_token_cap": 512 if not sources else max(len(s["token_ids"]) for s in sources),
                "sources": sources, "excluded_common_trigrams": [list(t) for t in sorted(excluded)],
                "fragments": fragments}
    return state, metadata
