import json
from types import SimpleNamespace

import pytest
import torch

from kv_dreaming.cache import KVState
from kv_dreaming import paired
from kv_dreaming.paired import PairConfig, masked_kl, run_pair, run_both, state_hash
from kv_dreaming.cli import paired_plan, sink_plan, parser


class Runtime:
    theta = 1e6
    inv_freq = torch.ones(1)
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: " ".join(map(str, tokens)))
    model = SimpleNamespace(config=SimpleNamespace(bos_token_id=1, max_position_embeddings=1024))

    def __init__(self, perturb=None, eos=False, follower_eos=False):
        self.calls = []
        self.perturb, self.eos, self.follower_eos = perturb, eos, follower_eos

    def step(self, token, state):
        self.calls.append((id(state), token))
        merged = max(state.counts, default=1) > 1
        zero = torch.zeros(1, 1, 1, 2)
        state.keys[0] = torch.cat((state.keys[0], zero), 2)
        state.values[0] = torch.cat((state.values[0], zero), 2)
        state.positions.append(float(state.next_position))
        state.next_position += 1
        state.counts.append(1)
        logits = torch.full((16,), -12.)
        logits[3], logits[4] = (1.0, -1.0) if not merged else (-1.0, 1.0)
        if self.eos or (self.follower_eos and merged):
            logits[2] = 1000.
        if self.perturb and len(self.calls) == 2:
            logits[8] = -12.0001
        return logits


@pytest.fixture(autouse=True)
def small_vocabulary(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(paired, "STOP_ID", 2)
    monkeypatch.setattr(paired, "DELIMITERS", {"blocked": 15})


def empty():
    return KVState([torch.empty(1, 1, 0, 2)], [torch.empty(1, 1, 0, 2)])


def config(**kwargs):
    return PairConfig(initial_state="empty", budget=12, recent=4, max_tokens=18, **kwargs)


def test_shadow_uses_leader_history_and_checks_first_reduction_step(tmp_path):
    runtime = Runtime(follower_eos=True)
    state = empty()
    original_hash = state_hash(state)
    summary = run_pair(runtime, state, config(), tmp_path / "pair", identity={})
    rows = [json.loads(s) for s in (tmp_path / "pair/steps.jsonl").read_text().splitlines()]
    assert summary["first_reduction_step"] == 12
    assert summary["pre_reduction_gate_checks"] == 13
    assert rows[12]["pre_reduction_gate"]["passed"] and not rows[12]["post_reduction"]
    assert rows[13]["pre_reduction_gate"] is None and rows[13]["post_reduction"]
    assert summary["post_reduction_steps"] == 5
    assert summary["stop_reason"] == "token_limit"
    assert any(r["follower_sampled_token_id"] == 2 for r in rows[13:])
    assert all(a[1] == b[1] for a, b in zip(runtime.calls[::2], runtime.calls[1::2]))
    assert all(row["leader_cache_length"] <= 12 and row["follower_cache_length"] <= 12 for row in rows)
    assert state_hash(state) == original_hash


def test_gate_failure_is_terminal_and_preserves_logits(tmp_path):
    with pytest.raises(RuntimeError, match="bitwise equality failed"):
        run_pair(Runtime(perturb=True), empty(), config(), tmp_path / "bad", identity={})
    summary = json.loads((tmp_path / "bad/summary.json").read_text())
    failure = json.loads((tmp_path / "bad/failure.json").read_text())
    assert summary["stop_reason"] == "error" and not summary["complete_comparison"]
    assert summary["generated_tokens"] == 0
    assert failure["step"] == 0 and failure["pre_reduction_gate"]["max_abs_diff"] > 0
    assert (tmp_path / "bad/failure-logits.pt").exists()


def test_both_directions_share_initial_state_and_keep_early_eos(tmp_path):
    state = empty()
    reports = run_both(Runtime(eos=True), state, config(), tmp_path / "both", identity={})
    assert set(reports) == {"evict", "merge"}
    for report in reports.values():
        assert report["stop_reason"] == "eos" and report["generated_tokens"] == 1
        assert report["mean_post_reduction_kl"] is None
        assert report["post_reduction_token_flip_rate"] is None
        assert report["trajectory_metrics"]["self_eos_step"] == 0
    manifests = [json.loads((tmp_path / f"both/{p}-leader/manifest.json").read_text()) for p in reports]
    assert manifests[0]["initial_cache_sha256"] == manifests[1]["initial_cache_sha256"] == state_hash(state)


def test_masked_kl_matches_analytic_distribution():
    a = torch.tensor([.75, .25, 1.0]).log()
    b = torch.tensor([.5, .5, 999.]).log()
    expected = .75 * torch.log(torch.tensor(1.5)) + .25 * torch.log(torch.tensor(.5))
    assert masked_kl(a, b, [2]) == pytest.approx(float(expected), abs=1e-7)
    assert masked_kl(a, a, [2]) == 0


def test_paired_runner_with_native_qwen_attention(runtime, tmp_path, monkeypatch):
    runtime.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: " ".join(map(str, tokens)))
    native_step = runtime.step

    def step_without_early_eos(token, state):
        logits = native_step(token, state)
        # Keep this mechanical integration check alive long enough to reduce.
        logits[2] = -1000.
        return logits

    monkeypatch.setattr(runtime, "step", step_without_early_eos)
    state = runtime.prefill(token_ids=list(range(3, 11)))
    before = state_hash(state)
    cfg = PairConfig(initial_state="soft", prefix_length=8, budget=12,
                     recent=4, max_tokens=24)
    summaries = run_both(runtime, state, cfg, tmp_path / "native", identity={})
    assert state_hash(state) == before
    for policy, summary in summaries.items():
        assert summary["stop_reason"] == "token_limit"
        assert summary["pre_reduction_gate_checks"] == 5
        assert summary["first_reduction_step"] == 4
        assert summary["post_reduction_steps"] == 19
        assert summary["mean_post_reduction_kl"] > 0
        rows = [json.loads(line) for line in
                (tmp_path / f"native/{policy}-leader/steps.jsonl").read_text().splitlines()]
        assert all(row["masked_kl_leader_to_follower"] == 0 for row in rows[:5])
        assert all(row["pre_reduction_gate"]["passed"] for row in rows[:5])
        assert all(row["leader_cache_length"] <= 12 and row["follower_cache_length"] <= 12
                   for row in rows)


def test_plan_and_cli_round_trip():
    plan = paired_plan()
    assert plan["count"] == len(plan["runs"]) == 16 and plan["paired_directions"] == 32
    assert len({json.dumps(r, sort_keys=True) for r in plan["runs"]}) == 16
    for row in plan["runs"]:
        argv = ["paired", "--output", "/tmp/not-created"]
        for key, value in row.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        args = parser().parse_args(argv)
        assert args.leader_policy == "both" and args.budget == 128 and args.recent == 64
        if args.initial_state == "random":
            assert args.alpha_k == args.alpha_v == .25


def test_sink_plan_and_cli_round_trip():
    plan = sink_plan()
    assert plan["count"] == len(plan["runs"]) == 8 and plan["paired_directions"] == 16
    assert plan["tag"] == "paired-4b-sink-v1"
    for row in plan["runs"]:
        argv = ["paired", "--output", "/tmp/not-created"]
        for key, value in row.items():
            argv.extend(["--" + key.replace("_", "-"), str(value)])
        args = parser().parse_args(argv)
        assert args.fragment_sink == "bos" and args.prefix_length == 65
        assert args.initial_state == "fragment" and args.leader_policy == "both"
        assert args.budget == 128 and args.recent == 64 and args.max_tokens == 2048
    default = parser().parse_args(["paired", "--initial-state", "fragment", "--output", "/tmp/not-created"])
    assert default.fragment_sink == "none"
