"""Frozen P1/P2/P3 comparison from retained paired artifacts; no inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from kv_dreaming.pair_metrics import fragment_pre_reduction, trajectory_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(root, mode, plan):
    expected = {(seed, beta, policy) for seed in (11, 23, 47, 89)
                for beta in (0., .25) for policy in ("evict", "merge")}
    rows, errors, manifests = {}, [], {}
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        cfg = manifest.get("config", {})
        if manifest.get("kind") != "paired_shadow" or cfg.get("initial_state") != "fragment":
            continue
        if cfg.get("fragment_sink", "none") != mode:
            raise ValueError(f"unexpected fragment mode in {path}")
        key = (cfg["initialization_seed"], cfg["fragment_beta"], cfg["policy"])
        if key not in expected or key in rows:
            raise ValueError(f"unexpected or duplicate condition in {path}")
        expected_cfg = {"prefix_length": 65 if mode == "bos" else 64, "budget": 128, "recent": 64,
                        "sampling_seed": 101, "max_tokens": 2048, "snapshot_every": 0, "max_seconds": 900.0}
        if any(cfg[k] != v for k, v in expected_cfg.items()):
            raise ValueError(f"configuration mismatch in {path}")
        if manifest["source_hashes"] != plan["source_hashes"]:
            raise ValueError(f"source hash mismatch in {path}")
        metadata = manifest["initialization_metadata"]
        if metadata["manifest_sha256"] != plan["source_manifest_sha256"]:
            raise ValueError(f"fixture manifest mismatch in {path}")
        summary_path, token_path = path.parent / "summary.json", path.parent / "tokens.json"
        if not summary_path.exists() or not token_path.exists():
            errors.append(f"incomplete artifacts: {path.parent}")
            continue
        summary, tokens = json.loads(summary_path.read_text()), json.loads(token_path.read_text())
        steps_path = path.parent / "steps.jsonl"
        steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
        first = next((r["step"] for r in steps if r["events"]["leader"] or r["events"]["follower"]), None)
        if first != summary["first_reduction_step"] or len(tokens) != summary["generated_tokens"]:
            errors.append(f"step/summary mismatch: {path.parent}")
        if first not in {None, 63 if mode == "bos" else 64}:
            errors.append(f"unexpected reduction boundary: {path.parent}")
        if summary["stop_reason"] not in {"eos", "token_limit"} or summary["gate_failure"] is not None:
            errors.append(f"incomplete or gate-failed direction: {path.parent}")
        p3 = fragment_pre_reduction(tokens, metadata, first)
        if "fragment_pre_reduction" in summary and p3 != summary["fragment_pre_reduction"]:
            errors.append(f"P3 summary mismatch: {path.parent}")
        if trajectory_metrics(tokens, 151643) != summary["trajectory_metrics"]:
            errors.append(f"trajectory metric mismatch: {path.parent}")
        rows[key] = {"run": str(path.parent), "seed": key[0], "beta": key[1], "policy": key[2],
                     "stop_reason": summary["stop_reason"], "generated_tokens": len(tokens),
                     "post_reduction_steps": summary["post_reduction_steps"],
                     "P1_no_reduction": first is None,
                     "P2_loop_flag": summary["trajectory_metrics"]["loop_onset_tokens"] is not None,
                     "P3_pre_reduction_trigram_hit": p3["direction_any_hit"], "P3_details": p3,
                     "artifact_sha256": {p.name: sha(p) for p in (path, summary_path, token_path, steps_path)}}
        manifests[key] = manifest
    missing = sorted(expected - rows.keys())
    result = {"fragment_sink": mode, "expected_directions": 16, "present_directions": len(rows),
              "complete": not missing and not errors, "missing_conditions": missing, "errors": errors,
              "counts": {metric: sum(r[metric] for r in rows.values()) for metric in
                         ("P1_no_reduction", "P2_loop_flag", "P3_pre_reduction_trigram_hit")},
              "stop_reasons": dict(Counter(r["stop_reason"] for r in rows.values())),
              "runs": [rows[key] for key in sorted(rows)]}
    return result, manifests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--baseline-plan", type=Path, default=Path("artifacts/paired-plan-reviewed.json"))
    p.add_argument("--sink-root", type=Path)
    p.add_argument("--sink-plan", type=Path, default=Path("artifacts/fragment-sink-plan-reviewed.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite comparison output")
    baseline, bm = cohort(args.baseline_root, "none", json.loads(args.baseline_plan.read_text()))
    result = {"schema": 1, "baseline": baseline, "baseline_plan_sha256": sha(args.baseline_plan),
              "comparison_script_sha256": sha(Path(__file__)), "sink": None, "primary_verdict": "BASELINE_ONLY",
              "note": "Counts use 16 directions from 8 initial caches; paired leader directions are not independent replicates. A negative intervention stops fragment starts, not all sink hypotheses."}
    if args.sink_root:
        sink, sm = cohort(args.sink_root, "bos", json.loads(args.sink_plan.read_text()))
        for key in bm.keys() & sm.keys():
            for field in ("identity", "calibration_sha256"):
                if bm[key][field] != sm[key][field]:
                    raise ValueError(f"matched {field} differs for {key}")
            signature = lambda m: [{k: f[k] for k in ("source_id", "source_sha256", "source_offset",
                           "original_positions", "token_ids", "eligible_trigrams")}
                           for f in m["initialization_metadata"]["fragments"]]
            if signature(bm[key]) != signature(sm[key]):
                raise ValueError(f"matched source fragments differ for {key}")
        result.update(sink=sink, sink_plan_sha256=sha(args.sink_plan))
        if not baseline["complete"] or not sink["complete"]:
            result["primary_verdict"] = "INCOMPLETE_NO_PRIMARY_VERDICT"
        else:
            a, b = baseline["counts"], sink["counts"]
            predictions = {"P1": b["P1_no_reduction"] < a["P1_no_reduction"],
                           "P2": b["P2_loop_flag"] < a["P2_loop_flag"],
                           "P3": b["P3_pre_reduction_trigram_hit"] > a["P3_pre_reduction_trigram_hit"]}
            result.update(predictions=predictions, primary_verdict=
                          "PASS_OPERATIONAL_CRITERIA" if all(predictions.values()) else "STOP_FRAGMENT_STARTS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baseline_counts": baseline["counts"], "baseline_complete": baseline["complete"],
                      "primary_verdict": result["primary_verdict"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
