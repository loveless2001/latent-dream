from types import SimpleNamespace

import pytest
import torch

from kv_dreaming.cache import KVState, merge_pair
from kv_dreaming.rope import rotate, rotate_half
from kv_dreaming.seeding import calibrate, initialize


@torch.inference_mode()
def fixture_calibration(runtime, tmp_path, monkeypatch):
    n = 12
    content = torch.arange(1 * 2 * n * 8, dtype=torch.float32).reshape(1, 2, n, 8) / 100
    values = content.square()
    positions = torch.arange(n)[None, :]
    cos, sin = runtime.model.model.rotary_emb(content, positions)
    keys = content * cos[:, None] + rotate_half(content) * sin[:, None]
    state = KVState([keys] * 2, [values] * 2, list(map(float, range(n))), [1] * n, n, n)
    monkeypatch.setattr(runtime, "prefill", lambda **kwargs: state)
    runtime.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(n)))
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("fixed fixture")
    return calibrate(runtime, fixture, token_count=n), content, values


def test_calibration_unrotates_and_excludes_initial_slots(runtime, tmp_path, monkeypatch):
    data, keys, values = fixture_calibration(runtime, tmp_path, monkeypatch)
    for prefix, source in (("key", keys), ("value", values)):
        body = source[:, :, 4:]
        torch.testing.assert_close(data[prefix + "_means"][0], body.mean(2, keepdim=True))
        torch.testing.assert_close(data[prefix + "_stds"][0], body.std(2, keepdim=True, correction=0))
    torch.testing.assert_close(data["real_mean_key_norms"][0], keys[:, :, 4:].norm(dim=-1).mean(2))


def test_random_keys_use_calibrated_content_without_renormalizing(runtime, tmp_path, monkeypatch):
    data, _, _ = fixture_calibration(runtime, tmp_path, monkeypatch)
    data["key_stds"] = [torch.zeros_like(x) for x in data["key_stds"]]
    for layer in runtime.model.model.layers:
        def forbidden(*args, **kwargs):
            raise AssertionError("calibrated keys must not pass through k_norm")
        monkeypatch.setattr(layer.self_attn.k_norm, "forward", forbidden)
    state = initialize(runtime, "random", calibration=data, prefix_length=8, alpha_k=1.5)
    content = data["key_means"][0].expand(1, 2, 8, 8) * 1.5
    cos, sin = runtime.model.model.rotary_emb(content, torch.arange(8)[None, :])
    expected = content * cos[:, None] + rotate_half(content) * sin[:, None]
    torch.testing.assert_close(state.keys[0], expected)
    old = dict(data)
    old.pop("schema")
    with pytest.raises(ValueError, match="schema"):
        initialize(runtime, "random", calibration=old)
    malformed = {**data, "key_stds": data["key_stds"][:1]}
    with pytest.raises(ValueError, match="layer count"):
        initialize(runtime, "random", calibration=malformed)


@torch.inference_mode()
def test_rotation_matches_model_at_long_fractional_positions(runtime):
    positions = [4097.0, 16447.0, 30000.5]
    x = torch.randn(1, runtime.kv_heads, 3, runtime.head_dim)
    cos, sin = runtime.model.model.rotary_emb(x, torch.tensor([positions]))
    expected = x * cos[:, None] + rotate_half(x) * sin[:, None]
    actual = rotate(x, positions, inv_freq=runtime.inv_freq)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(rotate(actual, positions, inv_freq=runtime.inv_freq, inverse=True),
                               x, atol=5e-7, rtol=2e-6)


def test_merge_norm_summary_handles_zero_keys(runtime):
    keys = torch.zeros(1, 2, 2, 8)
    state = KVState([keys], [keys.clone()], [5.0, 6.0], [1, 2], 7, 0)
    event = merge_pair(state, 0, runtime.theta, inv_freq=runtime.inv_freq)
    stats = event["key_norms"][0]
    assert stats["merged_key_norms"] == {"mean": 0.0, "min": 0.0, "max": 0.0}
    assert stats["merged_to_weighted_input_ratio"] == {"mean": 0.0, "min": 0.0, "max": 0.0}
