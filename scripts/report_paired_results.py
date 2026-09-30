"""Render artifact-derived paired results, retaining missing-data distinctions."""
import argparse
from collections import Counter
import json
from pathlib import Path


def write_report(directory, output):
    check = json.loads((directory / "consistency.json").read_text())
    recall = json.loads((directory / "recall.json").read_text())
    comparisons = recall["fragment_comparisons"]
    prefix = directory.name
    context_path = directory / "execution-context.json"
    context = json.loads(context_path.read_text()) if context_path.exists() else {"tag": "paired-4b-v1"}
    tag = context["tag"]
    observed = [r for r in comparisons if r["observed"]["recall"] is not None]
    adjusted = [r for r in comparisons if r["recall_minus_baseline"] is not None]
    null_reasons = Counter(r["observed"].get("unavailable_reason", "observed_but_no_eligible_baseline")
                          for r in comparisons if r["recall_minus_baseline"] is None)
    lines = ["# Paired 4B policy and fragment results", "",
             f"Artifact-derived report for `{tag}`. The original batch was launched by Claude; Codex handled collection and offline checks, with one conditional capacity fallback authorized in #lab 20629. The separate short audit group is excluded from these counts.", "",
             f"Launcher GPU selection: **{context.get('gpu', 'not recorded')}**. Calibration was retained from the original 4B/CUDA calibration on L40S. Hardware type is recorded from the launcher selection, not independently measured telemetry. "
             "The within-group equality gate remains mandatory on the execution device.", "",
             "| Count | Value |", "| --- | ---: |",
             f"| Planned directions | {check['planned_directions']} |",
             f"| Present direction summaries | {check['present_directions']} |",
             f"| Completed bounded trajectories (EOS/token cap) | {check['completed_bounded_trajectories']} |",
             f"| Directions with post-reduction samples | {check['directions_with_post_reduction_samples']} |",
             f"| Recorded gate failures | {check['gate_failure_directions']} |",
             f"| Passed equality checks in retained step logs | {check['checked_gate_steps']} |", "",
             "Stop reasons: " + json.dumps(check["stop_reasons"], sort_keys=True) + ".", "",
             f"All expected artifacts pass consistency checks: **{check['all_expected_artifacts_consistent']}**. "
             f"Missing directions: {len(check['missing_directions'])}; consistency discrepancies: {len(check['consistency_errors'])}.", "",
             "## Policy measurements", "",
             "M1/M2 start on zero-based step 65, after the first reduction at step 64. Each table mean weights eligible completed directions equally; it is not pooled over tokens. A direction ending before any post-reduction sample has null M1/M2. The two leader directions follow their own generated histories after divergence, so these figures do not rank output quality.", "",
             "| Start / leader | Directions | With post-reduction data | Generated-token range | Mean KL | Mean token-flip rate | Loop flags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda value: "null" if value is None else f"{value:.6g}"
    for name, row in check["conditions"].items():
        counts = row["generated_tokens"]
        lines.append(f"| {name} | {row['directions']} | {row['post_reduction_directions']} | {min(counts)}–{max(counts)} | "
                     f"{fmt(row['run_mean_kl'])} | {fmt(row['run_mean_flip_rate'])} | {sum(n is not None for n in row['loop_onset_tokens'])} |")
    lines += ["", "## Fragment recurrence", "",
              f"There are {len(comparisons)} fragment/boundary measurements. Raw recall is available for {len(observed)}; "
              f"recall minus the cross-run baseline is available for {len(adjusted)}. "
              f"Positive raw recall occurs in {sum(r['observed']['recall'] > 0 for r in observed)} available measurements.", "",
              "| Reason adjusted recall is unavailable | Measurements |", "| --- | ---: |"]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(null_reasons.items())]
    if adjusted:
        values = [r["recall_minus_baseline"] for r in adjusted]
        lines += ["", f"Available adjusted recall ranges from {min(values):.6g} to {max(values):.6g}. "
                  "These repeated fragment/boundary measurements are not independent replicates."]
    lines += ["", "Each comparison retains its eligible comparator count, exclusions, post-boundary window lengths, recurrence end steps, and baseline mean. "
              "Protected initial slots and prior generated text can retain source influence; lexical recurrence does not identify a causal memory mechanism. "
              "Unequal output lengths and the small set of initialization seeds limit interpretation.", "",
              "## Evidence and scope", "",
              f"- [Consistency checks and per-direction metrics]({prefix}/consistency.json)",
              f"- [Offline fragment baseline analysis]({prefix}/recall.json)",
              f"- [Raw manifests, tokens, text, and step logs]({prefix}/raw/)",
              f"- [Calibration metadata]({prefix}/calibration.json)",
              "- Frozen plan: `kv-dreaming/artifacts/paired-plan-reviewed.json`, SHA-256 `" + check["plan_sha256"] + "`.",
              "- Download inventories retain SHA-256 for local files and remote paths/sizes for terminal cache tensors. "
              f"Terminal cache tensors remain on `kv-dream-runs:/{tag}/`; they were not downloaded or independently inspected.",
              "- The checker recomputes summary arithmetic and verifies recorded gate values, configuration, provenance, teacher-forcing history, "
              "cache bounds, and cross-direction pre-reduction token/statistic equality. Passing gate logs attest to the runtime check; full per-step logit vectors are not retained for successful steps.",
              "- Collection and analysis execute no model inference. Any authorized queue-capacity fallback is recorded in `execution-context.json`; no failed trajectory is automatically retried. [Claude's independent results audit](claude-260930-0215-kv-dreaming-paired-results-audit.md) confirmed the report with zero issues."]
    if check["missing_directions"] or check["consistency_errors"]:
        lines += ["", "## Unresolved findings", ""]
        lines += ["- Missing: `" + value + "`." for value in check["missing_directions"]]
        lines += ["- " + value for value in check["consistency_errors"]]
    output.write_text("\n".join(lines) + "\n")
    return check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite report")
    write_report(args.directory, args.output)
