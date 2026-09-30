"""Check retained paired artifacts against the frozen plan; no model execution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite check output")
    plan = read(args.plan)
    errors, reports, missing, pair_checks = [], [], [], []
    calibration_hashes, identities = set(), set()
    expected_groups = set()

    def check(condition, message):
        if not condition:
            errors.append(message)

    for planned in plan["runs"]:
        name = f"{planned['initial_state']}-i{planned['initialization_seed']}-s{planned['sampling_seed']}"
        if planned["initial_state"] == "random":
            name += f"-a{planned['alpha_k']:g}"
        elif planned["initial_state"] == "fragment":
            name += f"-b{planned['fragment_beta']:g}"
        expected_groups.add(name)
        group_path = args.input_root / name
        group = read(group_path / "group.json") if (group_path / "group.json").exists() else None
        directions = {}
        for policy in ("evict", "merge"):
            path = group_path / f"{policy}-leader"
            label = f"{name}/{policy}-leader"
            if not (path / "summary.json").exists():
                missing.append(label)
                continue
            manifest, summary, tokens, draws = [read(path / f"{n}.json")
                                                for n in ("manifest", "summary", "tokens", "uniforms")]
            rows = [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
            directions[policy] = (manifest, rows, tokens, draws)
            cfg = {k: v for k, v in planned.items()
                   if k not in {"preset", "leader_policy", "fragment_sources"}}
            cfg["policy"] = policy
            check(manifest["config"] == cfg, label + ": configuration mismatch")
            check(manifest["source_hashes"] == plan["source_hashes"], label + ": source hash mismatch")
            check(manifest["follower_policy"] == ("merge" if policy == "evict" else "evict"), label + ": follower policy")
            check(group is not None and group["initial_cache_sha256"] == manifest["initial_cache_sha256"], label + ": group cache hash")
            identity = manifest["identity"]
            check(identity == {"model": "Qwen/Qwen3-4B-Base", "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
                               "dtype": "float32", "device": "cuda", "threads": 4,
                               "attention": "shared_explicit_eager"}, label + ": backend identity")
            identities.add(json.dumps(identity, sort_keys=True))
            calibration_hashes.add(manifest["calibration_sha256"])
            check(len(tokens) == len(rows) == summary["generated_tokens"], label + ": length mismatch")
            check(tokens == [r["token_id"] for r in rows], label + ": token log mismatch")
            check(summary["complete_comparison"] == (summary["stop_reason"] in {"eos", "token_limit"}), label + ": completion mismatch")
            if summary["stop_reason"] == "eos":
                check(bool(tokens) and tokens[-1] == 151643, label + ": invalid EOS")
            if summary["stop_reason"] == "token_limit":
                check(len(tokens) == cfg["max_tokens"], label + ": premature token cap")
            check(151643 not in tokens[:-1], label + ": continued after EOS")
            prior_reduction, first_reduction, gates = False, None, 0
            post, events = [], Counter()
            for step, row in enumerate(rows):
                check(row["step"] == step, label + ": nonsequential step")
                check(row["uniform"] == draws[step], label + ": uniform mismatch")
                check(row["input_token_id"] == (tokens[step - 1] if step else 151643), label + ": input history mismatch")
                check(row["post_reduction"] == prior_reduction, label + ": post-reduction timing mismatch")
                check(row["shared_uniform_token_flip"] == (row["token_id"] != row["follower_sampled_token_id"]), label + ": flip mismatch")
                if not prior_reduction:
                    gate = row["pre_reduction_gate"]
                    check(gate is not None and gate["passed"] and gate["max_abs_diff"] == 0, label + ": raw gate mismatch")
                    check(row["masked_kl_leader_to_follower"] == 0 and not row["shared_uniform_token_flip"], label + ": unequal pre-reduction distributions")
                    gates += 1
                else:
                    check(row["pre_reduction_gate"] is None, label + ": gate beyond boundary")
                    post.append(row)
                for role in ("leader", "follower"):
                    check(row[f"{role}_cache_length"] <= cfg["budget"], label + ": cache budget exceeded")
                    events[role] += len(row["events"][role])
                if row["events"]["leader"] or row["events"]["follower"]:
                    if first_reduction is None:
                        first_reduction = step
                    prior_reduction = True
            check(first_reduction == summary["first_reduction_step"], label + ": first reduction mismatch")
            initial_slots = 0 if cfg["initial_state"] == "empty" else cfg["prefix_length"]
            expected_first_reduction = cfg["budget"] - initial_slots
            check(first_reduction in {None, expected_first_reduction}, label + ": unexpected first reduction")
            check(gates == summary["pre_reduction_gate_checks"], label + ": gate count mismatch")
            check(len(post) == summary["post_reduction_steps"], label + ": post count mismatch")
            check({role: events[role] for role in ("leader", "follower")} == summary["reduction_events"], label + ": reduction event count mismatch")
            for key, values in (("mean_post_reduction_kl", [r["masked_kl_leader_to_follower"] for r in post]),
                                ("post_reduction_token_flip_rate", [r["shared_uniform_token_flip"] for r in post])):
                expected = sum(values) / len(values) if values else None
                actual = summary[key]
                check((actual is None and expected is None) or
                      (actual is not None and expected is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)), label + ": " + key)
            metadata = manifest.get("initialization_metadata")
            if cfg["initial_state"] == "fragment":
                check(metadata["manifest_sha256"] == plan["source_manifest_sha256"], label + ": fixture manifest mismatch")
                sources = {s["id"]: s for s in metadata["sources"]}
                grams = lambda ids: set(zip(ids, ids[1:], ids[2:]))
                excluded = set.intersection(*(grams(s["full_token_ids"]) for s in sources.values()))
                for f in metadata["fragments"]:
                    selected = sources[f["source_id"]]["token_ids"][f["source_offset"]:f["source_offset"] + 16]
                    check(selected == f["token_ids"], label + ": fragment token provenance")
                    check(grams(selected) - excluded == set(map(tuple, f["eligible_trigrams"])), label + ": eligible trigram mismatch")
            reports.append({"run": label, "condition": cfg["initial_state"] +
                            (f" beta={cfg['fragment_beta']}" if cfg["initial_state"] == "fragment" else ""),
                            "policy": policy, "initialization_seed": cfg["initialization_seed"],
                            **{k: summary[k] for k in ("stop_reason", "complete_comparison", "generated_tokens", "post_reduction_steps",
                               "pre_reduction_gate_checks", "first_reduction_step", "mean_post_reduction_kl", "post_reduction_token_flip_rate",
                               "trajectory_metrics", "reduction_events", "gate_failure", "elapsed_seconds")}})
        if len(directions) == 2:
            a, b = directions["evict"], directions["merge"]
            initial_slots = 0 if planned["initial_state"] == "empty" else planned["prefix_length"]
            common = min(planned["budget"] - initial_slots + 1, len(a[2]), len(b[2]))
            prefix_equal = a[2][:common] == b[2][:common]
            check(prefix_equal, name + ": cross-direction pre-reduction tokens differ")
            check(a[3] == b[3], name + ": cross-direction uniforms differ")
            check(a[0]["initial_cache_sha256"] == b[0]["initial_cache_sha256"], name + ": initial cache differs")
            stats_equal = all(x["leader_stats"] == y["leader_stats"] for x, y in zip(a[1][:common], b[1][:common]))
            check(stats_equal, name + ": cross-direction pre-reduction statistics differ")
            pair_checks.append({"group": name, "pre_reduction_tokens_compared": common,
                                "tokens_equal": prefix_equal, "statistics_equal": stats_equal})
    found_groups = {p.name for p in args.input_root.iterdir() if p.is_dir()}
    check(found_groups <= expected_groups, "unplanned groups present")
    check(len(calibration_hashes) <= 1 and None not in calibration_hashes, "calibration hashes differ or missing")
    calibration_path = args.input_root.parent / "calibration.json"
    if calibration_path.exists():
        calibration = read(calibration_path)
        check(calibration_hashes == {calibration["sha256"]}, "run calibration differs from retained calibration metadata")
        check(identities == {json.dumps(calibration["identity"], sort_keys=True)}, "run backend differs from retained calibration metadata")
    by_condition = defaultdict(list)
    for r in reports:
        by_condition[r["condition"] + "/" + r["policy"]].append(r)
    aggregates = {}
    for condition, rows in by_condition.items():
        valid = [r for r in rows if r["complete_comparison"] and r["post_reduction_steps"] > 0]
        aggregates[condition] = {"directions": len(rows), "post_reduction_directions": len(valid),
            "generated_tokens": [r["generated_tokens"] for r in rows],
            "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
            "run_mean_kl": sum(r["mean_post_reduction_kl"] for r in valid) / len(valid) if valid else None,
            "run_mean_flip_rate": sum(r["post_reduction_token_flip_rate"] for r in valid) / len(valid) if valid else None,
            "loop_onset_tokens": [r["trajectory_metrics"]["loop_onset_tokens"] for r in rows]}
    result = {"plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "planned_directions": plan["paired_directions"], "present_directions": len(reports),
              "completed_bounded_trajectories": sum(r["complete_comparison"] for r in reports),
              "directions_with_post_reduction_samples": sum(r["post_reduction_steps"] > 0 for r in reports),
              "stop_reasons": dict(Counter(r["stop_reason"] for r in reports)),
              "gate_failure_directions": sum(r["gate_failure"] is not None for r in reports),
              "checked_gate_steps": sum(r["pre_reduction_gate_checks"] for r in reports),
              "missing_directions": missing, "consistency_errors": errors,
              "all_expected_artifacts_consistent": not errors and not missing,
              "calibration_hashes": sorted(calibration_hashes, key=str), "identities": [json.loads(i) for i in sorted(identities)],
              "conditions": aggregates, "cross_direction_checks": pair_checks, "runs": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("present_directions", "completed_bounded_trajectories", "directions_with_post_reduction_samples",
                                            "stop_reasons", "gate_failure_directions", "checked_gate_steps", "consistency_errors", "missing_directions")}))


if __name__ == "__main__":
    main()
