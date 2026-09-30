"""One-runtime leader/follower policy comparison with a hard equality gate."""

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from .cache import reduce_cache
from .pair_metrics import FragmentTracker, trajectory_metrics
from .runner import RunConfig, source_hashes, write_json
from .runtime import DELIMITERS, STOP_ID
from .sampling import sample, uniforms


@dataclass(frozen=True)
class PairConfig(RunConfig):
    initial_state: str = "soft"
    budget: int = 128
    recent: int = 64
    max_tokens: int = 2048
    max_seconds: float = 900.0
    snapshot_every: int = 0

    def validate(self, runtime):
        super().validate(runtime)
        if self.snapshot_every:
            raise ValueError("paired runs save terminal states only; snapshot_every must be zero")


def state_hash(state):
    digest = hashlib.sha256()
    digest.update(json.dumps({"positions": state.positions, "counts": state.counts,
                              "next_position": state.next_position,
                              "bootstrap_position": state.bootstrap_position}, sort_keys=True).encode())
    for tensor in [*state.keys, *state.values]:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def masked_kl(leader, follower, blocked_ids):
    # Match the sampler's float32 log-softmax, then normalize in float64 as
    # its inverse-CDF path does. Exclude masked entries to avoid 0 * NaN.
    distributions = []
    for logits in (leader, follower):
        logits = logits.detach().to(device="cpu", dtype=torch.float32).clone()
        logits[list(blocked_ids)] = -torch.inf
        lp = logits.log_softmax(-1).double()
        distributions.append(lp - torch.logsumexp(lp, dim=-1))
    a, b = distributions
    allowed = torch.isfinite(a)
    kl = float((a[allowed].exp() * (a[allowed] - b[allowed])).sum())
    if not torch.isfinite(torch.tensor(kl)) or kl < -1e-10:
        raise FloatingPointError(f"invalid masked KL: {kl}")
    return max(0.0, kl)


def run_pair(runtime, initial_state, config: PairConfig, output: Path, *, identity,
             calibration_hash=None, initialization_metadata=None):
    config.validate(runtime)
    initial_state.validate(finite=True)
    expected_length = 0 if config.initial_state == "empty" else config.prefix_length
    if (initial_state.length != expected_length or initial_state.next_position != expected_length
            or initial_state.bootstrap_position != expected_length):
        raise ValueError("paired initial cache does not match the declared fresh prefix")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    leader, follower = initial_state.clone(), initial_state.clone()
    follower_policy = "merge" if config.policy == "evict" else "evict"
    leader_tracker = FragmentTracker(initialization_metadata, initial_state.length)
    follower_tracker = FragmentTracker(initialization_metadata, initial_state.length)
    draws = uniforms(config.sampling_seed, config.max_tokens)
    manifest = {"schema": 1, "kind": "paired_shadow", "config": asdict(config),
                "follower_policy": follower_policy, "identity": identity,
                "source_hashes": source_hashes(), "calibration_sha256": calibration_hash,
                "initialization_metadata": initialization_metadata,
                "initial_cache_sha256": state_hash(initial_state),
                "shared_uniforms": True, "follower_input": "leader-generated tokens only",
                "blocked_tokens": DELIMITERS, "stop_ids": [STOP_ID],
                "bootstrap_token": runtime.model.config.bos_token_id,
                "sampler": "inverse_cdf_token_id_order_cpu", "temperature": 1.0,
                "gate": "bitwise raw-logit equality until a previous reduction can affect this forward",
                "metric_scope": "post-reduction means start on the step AFTER the first reduction",
                "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")}}
    write_json(output / "manifest.json", manifest)
    write_json(output / "uniforms.json", draws)
    tokens, post_kls, flips, post_flips = [], [], [], []
    follower_entropies = []
    prior_reduction, first_reduction = False, None
    gate_checks, reductions = 0, {"leader": 0, "follower": 0}
    reason, error, failure = "token_limit", None, None
    next_token = runtime.model.config.bos_token_id
    start = time.monotonic()
    try:
        with (output / "steps.jsonl").open("x") as log, torch.inference_mode():
            for step, draw in enumerate(draws):
                if time.monotonic() - start >= config.max_seconds:
                    reason = "wall_time"
                    break
                input_token = next_token
                leader_logits = runtime.step(input_token, leader)
                follower_logits = runtime.step(input_token, follower)
                leader_tracker.append()
                follower_tracker.append()
                gate = None
                if not prior_reduction:
                    equal = (leader_logits.dtype == follower_logits.dtype
                             and leader_logits.shape == follower_logits.shape
                             and torch.equal(leader_logits.contiguous().view(torch.uint8),
                                             follower_logits.contiguous().view(torch.uint8)))
                    gate = {"passed": equal, "max_abs_diff": float((leader_logits - follower_logits).abs().max())}
                    if not equal:
                        failure = {"step": step, "input_token": input_token, "uniform": draw,
                                   "pre_reduction_gate": gate,
                                   "leader_cache_length": leader.length,
                                   "follower_cache_length": follower.length}
                        write_json(output / "failure.json", failure)
                        torch.save({"leader": leader_logits.cpu(), "follower": follower_logits.cpu()},
                                   output / "failure-logits.pt")
                        raise RuntimeError(f"pre-reduction bitwise equality failed at step {step}; max diff {gate['max_abs_diff']}")
                    gate_checks += 1
                next_token, leader_stats = sample(leader_logits, draw, DELIMITERS.values(), STOP_ID)
                follower_token, follower_stats = sample(follower_logits, draw, DELIMITERS.values(), STOP_ID)
                kl = masked_kl(leader_logits, follower_logits, DELIMITERS.values())
                flip = next_token != follower_token
                was_post_reduction = prior_reduction
                if was_post_reduction:
                    post_kls.append(kl)
                    post_flips.append(flip)
                flips.append(flip)
                follower_entropies.append(follower_stats["masked_entropy"])
                events = {}
                for role, state, tracker, policy in (
                        ("leader", leader, leader_tracker, config.policy),
                        ("follower", follower, follower_tracker, follower_policy)):
                    events[role] = reduce_cache(state, policy=policy, budget=config.budget,
                                               recent=config.recent, theta=runtime.theta,
                                               inv_freq=runtime.inv_freq)
                    tracker.reduce(events[role], step)
                    if len(tracker.origins) != state.length:
                        raise RuntimeError("fragment provenance/cache length mismatch")
                    reductions[role] += len(events[role])
                if events["leader"] or events["follower"]:
                    prior_reduction = True
                    if first_reduction is None:
                        first_reduction = step
                tokens.append(next_token)
                record = {"step": step, "input_token_id": input_token, "token_id": next_token,
                          "follower_sampled_token_id": follower_token, "uniform": draw,
                          "masked_kl_leader_to_follower": kl, "shared_uniform_token_flip": flip,
                          "post_reduction": was_post_reduction, "pre_reduction_gate": gate,
                          "leader_stats": leader_stats, "follower_stats": follower_stats,
                          "leader_cache_length": leader.length, "follower_cache_length": follower.length,
                          "absolute_position": leader.next_position - 1, "events": events,
                          "elapsed_seconds": time.monotonic() - start}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                if next_token == STOP_ID:
                    reason = "eos"
                    break
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as exc:
        reason, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        write_json(output / "tokens.json", tokens)
        (output / "text.txt").write_text(runtime.tokenizer.decode(tokens, skip_special_tokens=False,
                                                                   clean_up_tokenization_spaces=False))
        summary = {"stop_reason": reason, "error": error, "generated_tokens": len(tokens),
                   "complete_comparison": reason in {"eos", "token_limit"},
                   "elapsed_seconds": time.monotonic() - start, "first_reduction_step": first_reduction,
                   "pre_reduction_gate_checks": gate_checks, "gate_failure": failure,
                   "post_reduction_steps": len(post_kls),
                   "mean_post_reduction_kl": sum(post_kls) / len(post_kls) if post_kls else None,
                   "post_reduction_token_flip_rate": sum(post_flips) / len(post_flips) if post_flips else None,
                   "all_step_token_flip_rate": sum(flips) / len(flips) if flips else None,
                   "mean_follower_entropy": sum(follower_entropies) / len(follower_entropies) if follower_entropies else None,
                   "leader_cache_length": leader.length, "follower_cache_length": follower.length,
                   "reduction_events": reductions, "trajectory_metrics": trajectory_metrics(tokens, STOP_ID),
                   "fragment_boundaries": {"leader": leader_tracker.boundaries,
                                           "follower": follower_tracker.boundaries},
                   "fragment_recall": leader_tracker.recall(tokens)}
        try:
            leader.validate(finite=True)
            follower.validate(finite=True)
            torch.save({"schema": 1, "terminal": True, "leader": leader.payload(),
                        "follower": follower.payload(), "next_token": next_token,
                        "source_hashes": manifest["source_hashes"]}, output / "final-state.pt")
        except Exception as exc:
            summary["snapshot_error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "summary.json", summary)
    if error:
        raise RuntimeError(error)
    return summary


def run_both(runtime, initial_state, config, output, **kwargs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    initial_hash = state_hash(initial_state)
    write_json(output / "group.json", {"schema": 1, "kind": "paired_group",
                                      "initial_cache_sha256": initial_hash,
                                      "source_hashes": source_hashes(), "config": asdict(config)})
    for policy in ("evict", "merge"):
        reports[policy] = run_pair(runtime, initial_state, replace(config, policy=policy),
                                   output / f"{policy}-leader", **kwargs)
        if state_hash(initial_state) != initial_hash:
            raise RuntimeError("paired runner mutated the shared original initial cache")
        if reports[policy]["stop_reason"] not in {"eos", "token_limit"}:
            break
    write_json(output / "summary.json", reports)
    return reports
