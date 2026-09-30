"""Bounded local schema-2 initialization check and 128-token smoke."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from kv_dreaming.cache import merge_pair
from kv_dreaming.rope import rotate, rotate_half
from kv_dreaming.runner import RunConfig, run, source_hashes, write_json
from kv_dreaming.runtime import BASE_MODEL, BASE_REVISION, QwenRuntime
from kv_dreaming.seeding import initialize


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke-output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.smoke_output.exists():
        raise SystemExit("refusing to overwrite verification or smoke")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    runtime = QwenRuntime.load(local_files_only=True)
    calibration = torch.load(args.calibration, map_location="cpu", weights_only=True)
    identity = {"model": BASE_MODEL, "revision": BASE_REVISION, "dtype": "float32",
                "device": "cpu", "threads": 4, "attention": "shared_explicit_eager"}
    if calibration["identity"] != identity:
        raise ValueError("calibration identity mismatch")
    state = initialize(runtime, "random", seed=11, calibration=calibration)
    ratios = torch.cat([(key.norm(dim=-1).mean(dim=2) / real).flatten()
                        for key, real in zip(state.keys, calibration["real_mean_key_norms"])])
    positions = [4097.0, 16447.0, 30000.5]
    x = torch.randn(1, runtime.kv_heads, 3, runtime.head_dim,
                    generator=torch.Generator().manual_seed(43))
    cos, sin = runtime.model.model.rotary_emb(x, torch.tensor([positions]))
    expected = x * cos[:, None] + rotate_half(x) * sin[:, None]
    actual = rotate(x, positions, inv_freq=runtime.inv_freq)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    event = merge_pair(state.clone(), 4, runtime.theta, inv_freq=runtime.inv_freq)
    calibration_hash = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    report = {"identity": identity, "calibration_sha256": calibration_hash,
              "source_hashes": source_hashes(), "initialization_seed": 11,
              "random_to_real_mean_key_norm": {"min": float(ratios.min()),
                 "median": float(ratios.median()), "max": float(ratios.max()),
                 "per_layer_head": ratios.reshape(len(state.keys), -1).tolist()},
              "rope_positions": positions, "rope_max_error": float((actual - expected).abs().max()),
              "merge_event_json_bytes": len(json.dumps(event).encode())}
    report["smoke"] = run(runtime, state, RunConfig(), args.smoke_output, identity=identity,
                           calibration_hash=calibration_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "random_to_real_mean_key_norm"}))


if __name__ == "__main__":
    main()
