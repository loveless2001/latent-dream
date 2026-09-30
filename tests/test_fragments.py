import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kv_dreaming.fragments import initialize_fragment
from kv_dreaming.rope import rotate_half
from kv_dreaming.seeding import CALIBRATION_METHOD, CALIBRATION_SCHEMA
from kv_dreaming.paired import PairConfig, run_both, state_hash
from kv_dreaming import paired


def source_fixture(tmp_path, runtime):
    rows = []
    for i in range(4):
        ids = [1, 2, 3] + [4 + (j * (i + 1) + i) % 45 for j in range(61)]
        p = tmp_path / f"source-{i}.txt"
        p.write_text(" ".join(map(str, ids)))
        rows.append({"id": str(i), "file": p.name, "genre": str(i),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"schema": 1, "sources": rows,
                                   "fragment_length": 16, "source_token_cap": 64}))
    runtime.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(map(int, text.split())))
    return manifest


@pytest.mark.parametrize("sink", ["none", "bos"])
@torch.inference_mode()
def test_fragment_splice_rephases_hf_keys_and_keeps_values(runtime, tmp_path, sink):
    manifest = source_fixture(tmp_path, runtime)
    state, meta = initialize_fragment(runtime, manifest, seed=23, sink=sink)
    assert state.length == state.next_position == state.bootstrap_position == 64 + int(sink == "bos")
    assert len({f["source_id"] for f in meta["fragments"]}) == 4
    assert [1, 2, 3] in meta["excluded_common_trigrams"]
    for fragment in meta["fragments"]:
        source = next(s for s in meta["sources"] if s["id"] == fragment["source_id"])
        real = runtime.prefill(token_ids=source["token_ids"])
        offset = fragment["source_offset"]
        start = fragment["virtual_positions"][0]
        old_p = torch.tensor([fragment["original_positions"]])
        new_p = torch.tensor([fragment["virtual_positions"]])
        for k, v, actual_k, actual_v in zip(real.keys, real.values, state.keys, state.values):
            old = k[:, :, offset:offset + 16]
            cos, sin = runtime.model.model.rotary_emb(old, old_p)
            content = old * cos[:, None] - rotate_half(old) * sin[:, None]
            cos, sin = runtime.model.model.rotary_emb(content, new_p)
            expected = content * cos[:, None] + rotate_half(content) * sin[:, None]
            torch.testing.assert_close(actual_k[:, :, start:start + 16], expected)
            torch.testing.assert_close(actual_v[:, :, start:start + 16], v[:, :, offset:offset + 16], atol=0, rtol=0)
        assert [1, 2, 3] not in fragment["eligible_trigrams"]


def test_beta_changes_noise_without_changing_fragment_selection(runtime, tmp_path):
    manifest = source_fixture(tmp_path, runtime)
    calibration = calibration_fixture()
    a, ma = initialize_fragment(runtime, manifest, seed=11)
    b, mb = initialize_fragment(runtime, manifest, seed=11, beta=.25, calibration=calibration)
    c, mc = initialize_fragment(runtime, manifest, seed=11, beta=.25, calibration=calibration)
    assert ma["fragments"] == mb["fragments"] == mc["fragments"]
    for x, y in zip(b.keys + b.values, c.keys + c.values):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert not torch.equal(a.keys[0], b.keys[0])
    assert not torch.equal(a.values[0], b.values[0])
    with pytest.raises(ValueError, match="requires"):
        initialize_fragment(runtime, manifest, beta=.25)
    (tmp_path / "source-0.txt").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        initialize_fragment(runtime, manifest)


def calibration_fixture():
    return {"schema": CALIBRATION_SCHEMA, "method": CALIBRATION_METHOD,
                   **{k: [torch.ones(1, 2, 1, 8) * value for _ in range(2)]
                      for k, value in (("key_means", .1), ("key_stds", .2),
                                       ("value_means", .3), ("value_stds", .4))},
                   "embedding_mean": torch.zeros(32), "embedding_std": torch.ones(32)}


@pytest.mark.parametrize("beta", [0., .25])
def test_default_fragment_cache_matches_frozen_v1_bitwise(runtime, tmp_path, beta):
    fixture = Path(__file__).parent / "fixtures/fragments_v1.py"
    # Original paired-v1 frozen plan's fragments.py hash.
    expected = "042b1b54ada6e4747676e331f5a4bb137f506739b415cfe101ccd66bc6a47218"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("kv_dreaming._fragments_v1", fixture)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    manifest = source_fixture(tmp_path, runtime)
    kwargs = dict(seed=23, beta=beta, calibration=calibration_fixture())
    old, old_meta = legacy.initialize_fragment(runtime, manifest, **kwargs)
    current, current_meta = initialize_fragment(runtime, manifest, **kwargs)
    explicit, explicit_meta = initialize_fragment(runtime, manifest, sink="none", **kwargs)
    assert state_hash(old) == state_hash(current) == state_hash(explicit)
    assert old_meta == current_meta == explicit_meta


@pytest.mark.parametrize("beta", [0., .25])
def test_bos_slot_is_bitwise_prefill_and_never_noised(runtime, tmp_path, beta):
    manifest = source_fixture(tmp_path, runtime)
    bos = runtime.prefill(token_ids=[runtime.model.config.bos_token_id])
    state, meta = initialize_fragment(runtime, manifest, seed=11, beta=beta,
                                      calibration=calibration_fixture(), sink="bos")
    plain, plain_meta = initialize_fragment(runtime, manifest, seed=11, beta=beta,
                                            calibration=calibration_fixture())
    for actual, expected in zip(state.keys + state.values, bos.keys + bos.values):
        assert torch.equal(actual[:, :, :1].contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    assert meta["sink"] == {"kind": "bos", "token_id": 1, "virtual_position": 0,
                            "protected": True, "noise_applied": False}
    assert meta["fragments"][0]["protected_positions"] == [1, 2, 3]
    assert state.positions == list(map(float, range(65)))
    for shifted, original in zip(meta["fragments"], plain_meta["fragments"]):
        assert shifted["source_id"] == original["source_id"]
        assert shifted["source_offset"] == original["source_offset"]
        assert shifted["token_ids"] == original["token_ids"]
        assert shifted["eligible_trigrams"] == original["eligible_trigrams"]
        assert shifted["virtual_positions"] == [p + 1 for p in original["virtual_positions"]]
    for actual, expected in zip(state.values, plain.values):
        assert torch.equal(actual[:, :, 1:].contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def test_sink_pair_gate_and_fragment_boundary_with_native_attention(runtime, tmp_path, monkeypatch):
    manifest = source_fixture(tmp_path, runtime)
    runtime.tokenizer.decode = lambda tokens, **kwargs: " ".join(map(str, tokens))
    state, meta = initialize_fragment(runtime, manifest, sink="bos")
    native_step = runtime.step

    def step(token, cache):
        logits = native_step(token, cache)
        logits[2] = -1000.  # Mechanical check must reach a reduction.
        return logits

    monkeypatch.setattr(runtime, "step", step)
    monkeypatch.setattr(paired, "STOP_ID", 2)
    monkeypatch.setattr(paired, "DELIMITERS", {"blocked": 63})
    config = PairConfig(initial_state="fragment", fragment_sink="bos", prefix_length=65, max_tokens=66)
    results = run_both(runtime, state, config, tmp_path / "paired", identity={}, initialization_metadata=meta)
    for summary in results.values():
        assert summary["first_reduction_step"] == 63
        assert summary["pre_reduction_gate_checks"] == 64
        assert summary["post_reduction_steps"] == 2
        assert summary["fragment_pre_reduction"]["window_tokens"] == 64
        first = summary["fragment_boundaries"]["leader"]["0"]
        assert first["first_affected_step"] == 63 and first["protected_positions"] == [1, 2, 3]
        assert first["all_original_affected_step"] is None
    with pytest.raises(ValueError, match="65"):
        PairConfig(initial_state="fragment", fragment_sink="bos", prefix_length=64, max_tokens=66).validate(runtime)
    with pytest.raises(ValueError, match="fragment initial state"):
        PairConfig(initial_state="soft", fragment_sink="bos", prefix_length=65, max_tokens=66).validate(runtime)
