"""A resumable trajectory, with immutable output directories and raw logs."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from .cache import KVState, reduce_cache
from .runtime import DELIMITERS, STOP_ID
from .sampling import sample, uniforms


@dataclass(frozen=True)
class RunConfig:
    initial_state: str = "random"
    policy: str = "evict"
    prefix_length: int = 64
    budget: int = 512
    recent: int = 256
    initialization_seed: int = 11
    sampling_seed: int = 101
    max_tokens: int = 128
    max_seconds: float = 300.0
    snapshot_every: int = 128
    alpha_k: float = 1.0
    alpha_v: float = 1.0
    fragment_beta: float = 0.0
    fragment_sink: str = "none"

    def validate(self, runtime):
        if self.initial_state not in {"empty", "random", "soft", "fragment"} or self.policy not in {"evict", "merge"}:
            raise ValueError("invalid initial state or cache policy")
        if self.max_tokens < 1 or self.max_seconds <= 0 or self.snapshot_every < 0:
            raise ValueError("invalid execution limit")
        if self.budget < self.recent + 7 or self.recent < 1 or self.prefix_length < 4:
            raise ValueError("invalid cache limits")
        n = 0 if self.initial_state == "empty" else self.prefix_length
        if n > self.budget or n + self.max_tokens > runtime.model.config.max_position_embeddings:
            raise ValueError("prefix/budget or checkpoint position limit exceeded")
        if not all(torch.isfinite(torch.tensor(x)) for x in (self.alpha_k, self.alpha_v, self.fragment_beta, self.max_seconds)):
            raise ValueError("non-finite setting")
        if self.alpha_k < 0 or self.alpha_v < 0 or self.fragment_beta < 0:
            raise ValueError("noise strengths must be nonnegative")
        if self.fragment_sink not in {"none", "bos"}:
            raise ValueError("fragment sink must be none or bos")
        if self.fragment_sink != "none" and self.initial_state != "fragment":
            raise ValueError("a fragment sink requires a fragment initial state")
        if self.initial_state == "fragment" and self.prefix_length != 64 + int(self.fragment_sink == "bos"):
            raise ValueError("fragment prefix length must be 64 without a sink or 65 with a BOS sink")


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def snapshot(path, *, state, next_token, step, draws, tokens, config, identity, terminal,
             initialization_metadata=None):
    state.validate(finite=True)
    data = {"schema": 1, "state": state.payload(), "next_token": next_token,
            "step": step, "uniforms": draws, "tokens": tokens,
            "config": asdict(config), "identity": identity, "terminal": terminal,
            "source_hashes": source_hashes(), "initialization_metadata": initialization_metadata}
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def read_snapshot(path, *, runtime, identity):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema") != 1 or data["identity"] != identity:
        raise ValueError("snapshot model/backend identity mismatch")
    if data["source_hashes"] != source_hashes():
        raise ValueError("snapshot implementation changed")
    if data["terminal"]:
        raise ValueError("a completed trajectory cannot be resumed")
    if data["step"] != len(data["tokens"]) or data["step"] > len(data["uniforms"]):
        raise ValueError("invalid snapshot sampling position")
    state = KVState.from_payload(data["state"], device=runtime.device)
    return data, state


def run(runtime, state, config: RunConfig, output: Path, *, identity, calibration_hash=None,
        resume_data=None, initialization_metadata=None):
    config.validate(runtime)
    output.mkdir(parents=True, exist_ok=False)
    draws = uniforms(config.sampling_seed, config.max_tokens)
    tokens, step, next_token = [], 0, runtime.model.config.bos_token_id
    if resume_data is not None:
        if resume_data["config"] != asdict(config):
            raise ValueError("resume configuration cannot change")
        draws, tokens = resume_data["uniforms"], list(resume_data["tokens"])
        step, next_token = resume_data["step"], resume_data["next_token"]
        initialization_metadata = resume_data.get("initialization_metadata")
    manifest = {"schema": 1, "config": asdict(config), "identity": identity,
                "source_hashes": source_hashes(), "calibration_sha256": calibration_hash,
                "initialization_metadata": initialization_metadata,
                "blocked_tokens": DELIMITERS, "stop_ids": [STOP_ID],
                "bootstrap_token": runtime.model.config.bos_token_id,
                "temperature": 1.0, "sampler": "inverse_cdf_token_id_order",
                "attention_backend": "shared_explicit_eager", "resumed_at_step": step,
                "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")}}
    write_json(output / "manifest.json", manifest)
    start = time.monotonic()
    reason, error = "token_limit", None
    reductions = 0
    trace = output / "steps.jsonl"
    try:
        with trace.open("x") as log, torch.inference_mode():
            while step < config.max_tokens:
                if time.monotonic() - start >= config.max_seconds:
                    reason = "wall_time"
                    break
                position = state.next_position
                logits = runtime.step(next_token, state)
                transient_length = state.length
                next_token, stats = sample(logits, draws[step], DELIMITERS.values(), STOP_ID)
                events = reduce_cache(state, policy=config.policy, budget=config.budget,
                                      recent=config.recent, theta=runtime.theta,
                                      inv_freq=runtime.inv_freq)
                reductions += len(events)
                tokens.append(next_token)
                record = {"step": step, "token_id": next_token, "absolute_position": position,
                          "cache_length": state.length, "transient_cache_length": transient_length,
                          "other_control_token": (runtime.tokenizer.convert_ids_to_tokens(next_token)
                              if next_token in runtime.tokenizer.added_tokens_decoder
                              and next_token not in DELIMITERS.values() and next_token != STOP_ID else None),
                          "elapsed_seconds": time.monotonic() - start,
                          "events": events, **stats}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                step += 1
                if next_token == STOP_ID:
                    reason = "eos"
                    break
                if config.snapshot_every and step % config.snapshot_every == 0:
                    snapshot(output / f"snapshot-{step:06d}.pt", state=state, next_token=next_token,
                             step=step, draws=draws, tokens=tokens, config=config, identity=identity,
                             terminal=step == config.max_tokens,
                             initialization_metadata=initialization_metadata)
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as exc:
        reason, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.monotonic() - start
        (output / "tokens.json").write_text(json.dumps(tokens) + "\n")
        (output / "text.txt").write_text(runtime.tokenizer.decode(tokens, skip_special_tokens=False,
                                                                  clean_up_tokenization_spaces=False))
        summary = {"stop_reason": reason, "error": error, "generated_tokens": len(tokens),
                   "new_tokens": step - manifest["resumed_at_step"], "elapsed_seconds": elapsed,
                   "cache_length": state.length, "next_position": state.next_position,
                   "reduction_events_this_segment": reductions}
        # A runtime interruption can occur halfway through a forward pass.
        # Preserve raw evidence, but never mark such a partial cache resumable.
        try:
            snapshot(output / "final-state.pt", state=state, next_token=next_token, step=step,
                     draws=draws, tokens=tokens, config=config, identity=identity,
                     terminal=True, initialization_metadata=initialization_metadata)
        except Exception as exc:
            summary["snapshot_error"] = f"{type(exc).__name__}: {exc}"
        write_json(output / "summary.json", summary)
    if error:
        raise RuntimeError(error)
    return summary
