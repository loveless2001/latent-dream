import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch

from .diagnostic import continuation_kl
from .runner import RunConfig, read_snapshot, run, write_json, source_hashes
from .runtime import BASE_MODEL, BASE_REVISION, MODEL_PRESETS, QwenRuntime
from .seeding import CALIBRATION_SCHEMA, calibrate, initialize
from .fragments import initialize_fragment
from .paired import PairConfig, run_pair, run_both
from .pair_metrics import analyze_pairs


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description="Frozen-model generation from synthetic KV state")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("plan-matrix", "plan-pairs"):
        matrix = sub.add_parser(name, help="write a plan; execute nothing")
        matrix.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("analyze-pairs", help="compute offline fragment-recall references")
    analyze.add_argument("--input-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    for name in ("calibrate", "run", "paired", "diagnostic"):
        s = sub.add_parser(name)
        s.add_argument("--preset", choices=MODEL_PRESETS, default="qwen3-0.6b-base")
        s.add_argument("--model")
        s.add_argument("--revision")
        s.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
        s.add_argument("--device", default="cpu")
        s.add_argument("--threads", type=int, default=4)
        s.add_argument("--local-files-only", action="store_true")
        s.add_argument("--output", type=Path, required=True)
        if name == "calibrate":
            s.add_argument("--fixture", type=Path, default=Path("data/alice_excerpt.txt"))
            s.add_argument("--tokens", type=int, default=1024)
        elif name == "diagnostic":
            s.add_argument("--fixture", type=Path, default=Path("data/alice_excerpt.txt"))
            s.add_argument("--max-seconds", type=float, default=900)
        else:
            s.add_argument("--calibration", type=Path)
            if name == "run":
                s.add_argument("--resume", type=Path)
                s.add_argument("--policy", choices=("evict", "merge"), default="evict")
            else:
                s.add_argument("--leader-policy", choices=("evict", "merge", "both"), default="both")
                s.set_defaults(policy="evict", resume=None)
            s.add_argument("--initial-state", choices=("empty", "random", "soft", "fragment"),
                           default="soft" if name == "paired" else "random")
            s.add_argument("--fragment-sources", type=Path, default=Path("data/fragments/sources.json"))
            s.add_argument("--fragment-beta", type=float, default=0.0)
            for flag, default in (("prefix-length",64), ("budget",128 if name == "paired" else 512),
                                  ("recent",64 if name == "paired" else 256),
                                  ("initialization-seed",11), ("sampling-seed",101),
                                  ("max-tokens",2048 if name == "paired" else 128),
                                  ("snapshot-every",0 if name == "paired" else 128)):
                s.add_argument("--"+flag, type=int, default=default)
            s.add_argument("--max-seconds", type=float, default=900 if name == "paired" else 300)
            s.add_argument("--alpha-k", type=float, default=1.0)
            s.add_argument("--alpha-v", type=float, default=1.0)
    return p


def matrix_plan():
    rows = []
    for policy in ("evict", "merge"):
        for kind in ("empty", "random", "soft"):
            for seed in ([11] if kind == "empty" else [11, 23, 47, 89]):
                for stream in (101, 202):
                    cfg = RunConfig(initial_state=kind, policy=policy, initialization_seed=seed,
                                    sampling_seed=stream, max_tokens=2048, max_seconds=900)
                    rows.append(asdict(cfg))
    return {"status": "PLAN_ONLY", "runs": rows, "count": len(rows),
            "note": "Audit the harness before exploratory execution. No run is launched by this command."}


def paired_plan():
    rows = []
    for kind, alpha, beta in (("soft", 1.0, 0.0), ("random", 0.25, 0.0),
                              ("fragment", 1.0, 0.0), ("fragment", 1.0, 0.25)):
        for seed in (11, 23, 47, 89):
            row = asdict(PairConfig(initial_state=kind, initialization_seed=seed,
                                    alpha_k=alpha, alpha_v=alpha, fragment_beta=beta))
            row.pop("policy")
            row.update(leader_policy="both", preset="qwen3-4b-base",
                       fragment_sources="data/fragments/sources.json")
            rows.append(row)
    return {"status": "PLAN_ONLY", "command": "paired", "runs": rows,
            "count": 16, "paired_directions": 32,
            "source_manifest_sha256": file_hash("data/fragments/sources.json"),
            "source_hashes": source_hashes(),
            "note": "One initial cache and loaded runtime per group; evict-leader then merge-leader. Audit before execution."}


def main():
    args = parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    if args.command in {"plan-matrix", "plan-pairs", "analyze-pairs"}:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = (matrix_plan() if args.command == "plan-matrix" else
                  paired_plan() if args.command == "plan-pairs" else analyze_pairs(args.input_root))
        write_json(args.output, result)
        print(args.output)
        return
    if args.threads < 1:
        raise SystemExit("threads must be positive")
    if (args.model is None) != (args.revision is None):
        raise SystemExit("provide both --model and --revision, or use --preset")
    if args.model is None:
        args.model, args.revision = MODEL_PRESETS[args.preset]
    calibration, calibration_hash = None, None
    if args.command in {"run", "paired"} and args.calibration:
        calibration = torch.load(args.calibration, map_location="cpu", weights_only=True)
        if calibration.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError("calibration schema mismatch; recalibrate this model")
        calibration_hash = file_hash(args.calibration)
    if str(args.device).startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    runtime = QwenRuntime.load(model_id=args.model, revision=args.revision, dtype=args.dtype,
                               device=args.device, local_files_only=args.local_files_only)
    identity = {"model": args.model, "revision": args.revision, "dtype": args.dtype,
                "device": args.device, "threads": args.threads,
                "attention": "shared_explicit_eager"}
    if args.command == "calibrate":
        data = calibrate(runtime, args.fixture, token_count=args.tokens)
        data["identity"] = identity
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, args.output)
        write_json(args.output.with_suffix(".json"),
                   {"schema": data["schema"], "method": data["method"],
                    "identity": identity, "sha256": file_hash(args.output),
                    "fixture_sha256": data["fixture_sha256"], "token_count": args.tokens,
                    "excluded_leading_tokens": 4, "source_hashes": source_hashes()})
        print(args.output)
    elif args.command == "diagnostic":
        ids = runtime.tokenizer.encode(args.fixture.read_text(), add_special_tokens=False)
        result = continuation_kl(runtime, ids, max_seconds=args.max_seconds)
        result.update(identity=identity, fixture_sha256=file_hash(args.fixture),
                      source_hashes=source_hashes())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({k:v for k,v in result.items() if k != "policies"}))
    else:
        config_type = PairConfig if args.command == "paired" else RunConfig
        if args.command == "paired" and args.leader_policy != "both":
            args.policy = args.leader_policy
        cfg = config_type(**{key: getattr(args, key) for key in config_type.__dataclass_fields__})
        cfg.validate(runtime)
        if calibration is not None:
            if calibration["identity"] != identity:
                raise ValueError("calibration checkpoint/backend identity mismatch")
        initialization_metadata = None
        if args.resume:
            resumed, state = read_snapshot(args.resume, runtime=runtime, identity=identity)
            cfg = RunConfig(**resumed["config"])
        else:
            resumed = None
            if cfg.initial_state == "fragment":
                state, initialization_metadata = initialize_fragment(
                    runtime, args.fragment_sources, seed=cfg.initialization_seed,
                    beta=cfg.fragment_beta, calibration=calibration)
            else:
                state = initialize(runtime, cfg.initial_state, seed=cfg.initialization_seed,
                                   prefix_length=cfg.prefix_length, calibration=calibration,
                                   alpha_k=cfg.alpha_k, alpha_v=cfg.alpha_v)
        kwargs = dict(identity=identity, calibration_hash=calibration_hash,
                      initialization_metadata=initialization_metadata)
        if args.command == "paired":
            runner = run_both if args.leader_policy == "both" else run_pair
            result = runner(runtime, state, cfg, args.output, **kwargs)
        else:
            result = run(runtime, state, cfg, args.output, resume_data=resumed, **kwargs)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
