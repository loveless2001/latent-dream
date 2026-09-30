"""Descriptive paired metrics and lexical recall with explicit censoring."""

import json
from pathlib import Path

from .fragments import trigrams


def trajectory_metrics(tokens, stop_id):
    onset = next((n for n in range(128, len(tokens) + 1, 32)
                  if len(set(tokens[n - 128:n])) / 128 < 0.12), None)
    bigrams = list(zip(tokens, tokens[1:]))
    return {"loop_onset_tokens": onset,
            "loop_definition": "first 128-token window with unique fraction < 0.12, stride 32",
            "self_eos_step": len(tokens) - 1 if tokens and tokens[-1] == stop_id else None,
            "distinct_bigram_ratio": len(set(bigrams)) / len(bigrams) if bigrams else None}


def recall_after(tokens, eligible, boundary):
    eligible = set(map(tuple, eligible))
    result = {"boundary_step": boundary, "eligible_trigrams": len(eligible),
              "post_boundary_tokens": 0 if boundary is None else max(0, len(tokens) - boundary - 1),
              "recall": None, "last_recurrence_step": None, "recurrences": []}
    if boundary is None:
        result["unavailable_reason"] = "boundary_not_reached"
        return result
    if not eligible:
        result["unavailable_reason"] = "no_eligible_trigrams"
        return result
    if result["post_boundary_tokens"] < 3:
        result["unavailable_reason"] = "no_post_boundary_trigram_opportunity"
        return result
    last = {}
    for i in range(boundary + 1, len(tokens) - 2):
        tri = tuple(tokens[i:i + 3])
        if tri in eligible:
            last[tri] = i + 2
    result["recall"] = len(last) / len(eligible)
    result["last_recurrence_step"] = max(last.values()) if last else None
    result["recurrences"] = [{"trigram": list(t), "last_step": last.get(t)} for t in sorted(eligible)]
    return result


class FragmentTracker:
    """Track which original synthetic slots any reduction has touched."""
    def __init__(self, metadata, length):
        self.fragments = (metadata or {}).get("fragments", [])
        self.origins = [{i} for i in range(length)]
        self.affected = set()
        self.boundaries = {str(f["id"]): {"first_affected_step": None,
                          "all_unprotected_affected_step": None,
                          "all_original_affected_step": None,
                          "protected_positions": f["protected_positions"]} for f in self.fragments}

    def append(self):
        self.origins.append(set())

    def reduce(self, events, step):
        for event in events:
            i = event["index"]
            if event["kind"] == "evict":
                self.affected.update(self.origins.pop(i))
            else:
                merged = self.origins[i] | self.origins[i + 1]
                self.affected.update(merged)
                self.origins[i:i + 2] = [merged]
        for fragment in self.fragments:
            positions = set(fragment["virtual_positions"])
            unprotected = positions - set(fragment["protected_positions"])
            b = self.boundaries[str(fragment["id"])]
            if positions & self.affected and b["first_affected_step"] is None:
                b["first_affected_step"] = step
            if unprotected and unprotected <= self.affected and b["all_unprotected_affected_step"] is None:
                b["all_unprotected_affected_step"] = step
            if positions <= self.affected and b["all_original_affected_step"] is None:
                b["all_original_affected_step"] = step

    def recall(self, tokens):
        return [{"fragment_id": f["id"], "source_id": f["source_id"],
                 "protected_positions": f["protected_positions"],
                 "boundaries": {key: recall_after(tokens, f["eligible_trigrams"], b[key])
                                for key in ("first_affected_step", "all_unprotected_affected_step",
                                            "all_original_affected_step")}}
                for f in self.fragments for b in [self.boundaries[str(f["id"])]]]


def analyze_pairs(input_root):
    runs = []
    for path in sorted(Path(input_root).rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("kind") != "paired_shadow":
            continue
        summary = json.loads((path.parent / "summary.json").read_text())
        tokens = json.loads((path.parent / "tokens.json").read_text())
        runs.append({"path": str(path.parent), "manifest": manifest,
                     "summary": summary, "tokens": tokens})
    if not runs:
        raise ValueError("no paired runs found")
    output = []
    for target in runs:
        tm = target["manifest"]
        metadata = tm.get("initialization_metadata") or {}
        for fragment, recalls in zip(metadata.get("fragments", []),
                                     target["summary"].get("fragment_recall", [])):
            eligible = set(map(tuple, fragment["eligible_trigrams"]))
            for boundary_name, observed in recalls["boundaries"].items():
                baseline, exclusions = [], []
                for other in runs:
                    om = other["manifest"]
                    ometa = om.get("initialization_metadata") or {}
                    if other is target or ometa.get("kind") != "fragment":
                        continue
                    reason = None
                    if (om["identity"] != tm["identity"] or om["source_hashes"] != tm["source_hashes"]
                            or om["calibration_sha256"] != tm["calibration_sha256"]
                            or ometa.get("manifest_sha256") != metadata.get("manifest_sha256")):
                        reason = "provenance_mismatch"
                    elif any(om["config"][k] != tm["config"][k] for k in
                             ("policy", "fragment_beta", "budget", "recent", "max_tokens", "sampling_seed")):
                        reason = "condition_mismatch"
                    elif om["config"]["initialization_seed"] == tm["config"]["initialization_seed"]:
                        reason = "same_initialization_seed"
                    elif other["summary"]["stop_reason"] not in {"eos", "token_limit"}:
                        reason = "incomplete_comparator"
                    elif eligible & set().union(*(trigrams(f["token_ids"]) for f in ometa["fragments"])):
                        reason = "target_trigram_present_in_comparator_seed"
                    comparison = recall_after(other["tokens"], eligible, observed["boundary_step"])
                    if reason is None and comparison["recall"] is None:
                        reason = comparison["unavailable_reason"]
                    if reason:
                        exclusions.append({"run": other["path"], "reason": reason})
                    else:
                        baseline.append({"run": other["path"], **comparison})
                mean = sum(b["recall"] for b in baseline) / len(baseline) if baseline else None
                valid_target = target["summary"]["stop_reason"] in {"eos", "token_limit"}
                output.append({"run": target["path"], "fragment_id": fragment["id"],
                               "boundary": boundary_name, "observed": observed,
                               "eligible_comparators": len(baseline), "baseline": baseline,
                               "excluded_comparators": exclusions, "baseline_mean_recall": mean,
                               "recall_minus_baseline": observed["recall"] - mean
                               if valid_target and observed["recall"] is not None and mean is not None else None,
                               "target_complete": valid_target})
    return {"schema": 1, "run_count": len(runs), "fragment_comparisons": output,
            "note": "Descriptive cross-run reference; windows start after the target boundary. Available lengths may differ. No significance or causal-memory claim."}
