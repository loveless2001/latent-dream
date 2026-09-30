"""Separate stdlib replay of fragment boundaries and token-trigram measurements."""
from collections import Counter
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
load = lambda p: json.loads(p.read_text())
grams = lambda ids: set(zip(ids, ids[1:], ids[2:]))
errors, runs = [], {}
event_count = 0


def check(ok, message):
    if not ok:
        errors.append(message)


def recall(tokens, eligible, boundary):
    if boundary is None or not eligible or len(tokens) - boundary - 1 < 3:
        return None, None
    occurrences = [(i + 2, tuple(tokens[i:i + 3])) for i in range(boundary + 1, len(tokens) - 2)
                   if tuple(tokens[i:i + 3]) in eligible]
    return len({tri for _, tri in occurrences}) / len(eligible), max((i for i, _ in occurrences), default=None)


for path in sorted((base / "raw").glob("*/*-leader/manifest.json")):
    manifest = load(path)
    meta = manifest.get("initialization_metadata") or {}
    if meta.get("kind") != "fragment":
        continue
    label = str(path.parent)
    summary, tokens = load(path.parent / "summary.json"), load(path.parent / "tokens.json")
    fragments = meta["fragments"]
    prefix_length = manifest["config"]["prefix_length"]
    histories = {role: [{i} for i in range(prefix_length)] for role in ("leader", "follower")}
    affected = {role: set() for role in histories}
    derived = {role: {str(f["id"]): {"first_affected_step": None, "all_unprotected_affected_step": None,
                    "all_original_affected_step": None, "protected_positions": f["protected_positions"]}
                     for f in fragments} for role in histories}
    for line in (path.parent / "steps.jsonl").read_text().splitlines():
        row = json.loads(line)
        step = row["step"]
        for role, nodes in histories.items():
            nodes.append({prefix_length + step})
            for event in row["events"][role]:
                event_count += 1
                i = event["index"]
                indices = [i] if event["kind"] == "evict" else [i, i + 1]
                check(all(j < len(nodes) - manifest["config"]["recent"] for j in indices), label + ": touched recent tail")
                touched = set().union(*(nodes[j] for j in indices))
                check(not touched.intersection({0, 1, 2, 3, prefix_length}), label + ": touched pinned initial/bootstrap slot")
                affected[role].update(touched)
                if event["kind"] == "evict":
                    nodes.pop(i)
                else:
                    nodes[i:i + 2] = [touched]
            check(len(nodes) == row[f"{role}_cache_length"], label + ": replay cache length")
            for fragment in fragments:
                positions = set(fragment["virtual_positions"])
                unprotected = positions - set(fragment["protected_positions"])
                boundary = derived[role][str(fragment["id"])]
                conditions = {"first_affected_step": bool(positions & affected[role]),
                              "all_unprotected_affected_step": bool(unprotected) and unprotected <= affected[role],
                              "all_original_affected_step": positions <= affected[role]}
                for key, passed in conditions.items():
                    if passed and boundary[key] is None:
                        boundary[key] = step
    check(derived == summary["fragment_boundaries"], label + ": fragment boundary mismatch")
    runs[label] = {"manifest": manifest, "summary": summary, "tokens": tokens, "fragments": fragments}

analysis = load(base / "recall.json")
availability = Counter()
for comparison in analysis["fragment_comparisons"]:
    label = comparison["run"]
    target = runs[label]
    fragment = next(f for f in target["fragments"] if f["id"] == comparison["fragment_id"])
    eligible = set(map(tuple, fragment["eligible_trigrams"]))
    boundary = target["summary"]["fragment_boundaries"]["leader"][str(fragment["id"])][comparison["boundary"]]
    observed, last = recall(target["tokens"], eligible, boundary)
    check(observed == comparison["observed"]["recall"], label + ": observed recall mismatch")
    check(last == comparison["observed"]["last_recurrence_step"], label + ": last recurrence mismatch")
    expected_baseline = {}
    tm = target["manifest"]
    for other_label, other in runs.items():
        om = other["manifest"]
        if om["config"]["initialization_seed"] == tm["config"]["initialization_seed"]:
            continue
        if any(om["config"][k] != tm["config"][k] for k in ("policy", "fragment_beta", "budget", "recent", "max_tokens", "sampling_seed", "fragment_sink", "prefix_length")):
            continue
        if any(om[k] != tm[k] for k in ("identity", "source_hashes", "calibration_sha256")):
            continue
        if om["initialization_metadata"]["manifest_sha256"] != tm["initialization_metadata"]["manifest_sha256"]:
            continue
        if other["summary"]["stop_reason"] not in {"eos", "token_limit"}:
            continue
        if eligible & set().union(*(grams(f["token_ids"]) for f in other["fragments"])):
            continue
        value, _ = recall(other["tokens"], eligible, boundary)
        if value is not None:
            expected_baseline[other_label] = value
    check(expected_baseline == {row["run"]: row["recall"] for row in comparison["baseline"]}, label + ": comparator selection/recall mismatch")
    mean = sum(expected_baseline.values()) / len(expected_baseline) if expected_baseline else None
    check(mean == comparison["baseline_mean_recall"], label + ": baseline mean mismatch")
    adjusted = observed - mean if observed is not None and mean is not None else None
    check(adjusted == comparison["recall_minus_baseline"], label + ": adjusted recall mismatch")
    availability["observed_available"] += observed is not None
    availability["adjusted_available"] += adjusted is not None
    availability["observed_positive"] += observed is not None and observed > 0

result = {"fragment_directions": len(runs), "replayed_reduction_events": event_count,
          "fragment_boundary_measurements": len(analysis["fragment_comparisons"]), **availability,
          "errors": errors, "passed": not errors,
          "recall_json_sha256": hashlib.sha256((base / "recall.json").read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "scope": "Separate event/token replay; no model execution or cache tensor inspection."}
output = base / "fragment-crosscheck.json"
if output.exists():
    raise SystemExit("refusing to overwrite fragment crosscheck")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
