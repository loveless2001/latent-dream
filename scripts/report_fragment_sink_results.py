"""Render the frozen BOS intervention comparison from completed artifacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def write_report(directory, output):
    load = lambda name: json.loads((directory / name).read_text())
    comparison = load("comparison.json")
    check = load("consistency.json")
    replay = load("fragment-crosscheck.json")
    recall = load("recall.json")["fragment_comparisons"]
    if not (comparison["baseline"]["complete"] and comparison["sink"]["complete"]
            and check["all_expected_artifacts_consistent"] and replay["passed"]):
        raise ValueError("complete, consistent artifacts are required for the final report")
    if check["present_directions"] != 16 or check["completed_bounded_trajectories"] != 16:
        raise ValueError("expected all sixteen completed sink directions")
    if output.exists():
        raise ValueError("refusing to overwrite report")
    baseline, sink = comparison["baseline"], comparison["sink"]
    prefix = directory.name
    lines = ["# BOS fragment-start follow-up results", "",
             f"The frozen matched comparison returns **{comparison['primary_verdict']}** for `paired-4b-sink-v1`.", "",
             "All sixteen BOS directions completed. The comparison uses the sixteen retained no-sink fragment directions, "
             "matched on seed, beta, leader policy, model/backend/calibration, source text, fragment token IDs, and source offsets. "
             "Each cohort contains eight initial-cache groups; the two leader directions are not independent replicates. "
             "Claude launched the audited batch on L40S; Codex collected artifacts and ran offline checks. "
             "GPU type follows the launch record rather than independently measured telemetry.", "",
             "| Preregistered outcome | No sink | BOS | Required change | Passed |",
             "| --- | ---: | ---: | --- | --- |"]
    for key, metric, label, requirement in [
            ("P1", "P1_no_reduction", "No reduction reached", "Strict decrease"),
            ("P2", "P2_loop_flag", "Low-diversity loop flag", "Strict decrease"),
            ("P3", "P3_pre_reduction_trigram_hit", "Pre-policy fragment-trigram hit", "Strict increase")]:
        lines.append(f"| {key}: {label} | {baseline['counts'][metric]} / 16 | {sink['counts'][metric]} / 16 | "
                     f"{requirement} | {comparison['predictions'][key]} |")
    if comparison["primary_verdict"] == "STOP_FRAGMENT_STARTS":
        lines += ["", "The agreed rule requires all three strict improvements. This complete experiment fails that rule, "
                  "so stop this fragment-start line under the frozen design. The result does not reject every attention-sink hypothesis. "
                  "Any narrower gains below remain descriptive and do not waive the stop rule."]
    else:
        lines += ["", "All three preregistered operational criteria passed. This supports the initializer in this bounded setting; "
                  "it does not establish output quality, a unique attention-sink mechanism, or policy superiority."]
    lines += ["", "P1 counts trajectories with no reduction event, including early EOS. P2 uses the existing 128-token "
              "window, unique-token fraction below 0.12, and stride 32. P3 requires an eligible selected-fragment trigram wholly "
              "inside samples zero through the first reduction step inclusive, or the entire output for earlier EOS. "
              "Outputs shorter than three tokens count as no hit and retain an opportunity flag.", "",
              "P3 hits occur in both leader directions of seed 23, beta 0.25 without BOS, and both leader directions of "
              "seed 47, beta 0.25 with BOS. The matched aggregate therefore ties despite a change in which seed supplies the hits.", "",
              "| Availability and validation | No sink | BOS |", "| --- | ---: | ---: |",
              f"| Directions with post-reduction samples | {sum(r['post_reduction_steps'] > 0 for r in baseline['runs'])} | {check['directions_with_post_reduction_samples']} |",
              f"| Post-reduction samples | {sum(r['post_reduction_steps'] for r in baseline['runs'])} | {sum(r['post_reduction_steps'] for r in sink['runs'])} |",
              f"| P3 windows shorter than three samples | {sum(r['P3_details']['window_tokens'] < 3 for r in baseline['runs'])} | {sum(r['P3_details']['window_tokens'] < 3 for r in sink['runs'])} |", "",
              "BOS stop reasons: " + json.dumps(check["stop_reasons"], sort_keys=True) + ". "
              f"All {check['checked_gate_steps']} recorded pre-reduction equality checks pass with zero raw-logit difference. "
              f"All {len(check['cross_direction_checks'])} cross-direction prefix comparisons pass. "
              f"Missing directions: {len(check['missing_directions'])}; consistency discrepancies: {len(check['consistency_errors'])}; "
              f"gate-failed directions: {check['gate_failure_directions']}.", "",
              "The first reduction occurs at zero-based step 63 with BOS, versus 64 without it. Policy-sensitive samples "
              "therefore begin at steps 64 and 65 respectively. Initial slots 0–3 and bootstrap position 65 remain protected "
              "in the BOS variant. The independent event replay checks the protected slots, recent tail, and fragment boundaries.", "",
              "## Descriptive policy measurements", "",
              "Each mean below weights directions with post-reduction data equally. The leader directions generate different "
              "histories after divergence; KL and hypothetical shared-uniform token flips do not rank output quality.", "",
              "| Beta / leader | Directions with post-reduction data | Generated-token range | Mean KL | Mean token-flip rate | Loop flags |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda x: "null" if x is None else f"{x:.6g}"
    for name, row in check["conditions"].items():
        tokens = row["generated_tokens"]
        lines.append(f"| {name} | {row['post_reduction_directions']} / {row['directions']} | {min(tokens)}–{max(tokens)} | "
                     f"{fmt(row['run_mean_kl'])} | {fmt(row['run_mean_flip_rate'])} | {sum(x is not None for x in row['loop_onset_tokens'])} |")
    observed = [r for r in recall if r["observed"]["recall"] is not None]
    adjusted = [r for r in recall if r["recall_minus_baseline"] is not None]
    positive = [r for r in observed if r["observed"]["recall"] > 0]
    positive_targets = {(r["run"], r["fragment_id"]) for r in positive}
    positive_runs = {r["run"] for r in positive}
    positive_groups = sorted({(r["seed"], r["beta"]) for r in sink["runs"] if r["run"] in positive_runs})
    nulls = Counter(r["observed"].get("unavailable_reason", "observed_but_no_eligible_baseline")
                    for r in recall if r["recall_minus_baseline"] is None)
    lines += ["", "## Post-boundary fragment recurrence (M4)", "",
              f"Of {len(recall)} fragment/boundary measurements, raw recall is available for {len(observed)}, "
              f"and recall minus the matched cross-seed baseline is available for {len(adjusted)}. "
              f"There are {len(positive)} positive raw-recall measurements, spanning {len(positive_targets)} fragment/direction "
              f"combinations in {len(positive_runs)} directions. Their (seed, beta) groups are {positive_groups}. "
              "Multiple boundary definitions reuse the same fragment trajectory, so the count is not a count of independent successes.", "",
              "| Reason adjusted recall is unavailable | Measurements |", "| --- | ---: |"]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(nulls.items())]
    if adjusted:
        values = [r["recall_minus_baseline"] for r in adjusted]
        lines += ["", f"Available adjusted recall ranges from {min(values):.6g} to {max(values):.6g}."]
    lines += ["", f"A separate standard-library replay checked all {replay['replayed_reduction_events']} reduction events, "
              f"all {replay['fragment_boundary_measurements']} fragment/boundary measurements, comparator eligibility, and baseline arithmetic "
              f"with {len(replay['errors'])} discrepancies. Baselines match sink mode and prefix length. "
              "Protected fragment slots, earlier generated text, unequal output lengths, and the small set of seeds limit interpretation. "
              "These repeated boundary measurements are not independent replicates; lexical recurrence does not identify causal memory."]
    diagnostic = directory.parent / "claude-260930-0215-kv-dreaming-paired-results-audit/slot0-attention-4b.json"
    if diagnostic.exists():
        diagnostic_rows = json.loads(diagnostic.read_text())
        if len(diagnostic_rows) != 28 or any(r["replica_max_logit_diff"] != 0 for r in diagnostic_rows):
            raise ValueError("unexpected diagnostic coverage or replica mismatch")
        lines += ["", "## Separate bootstrap attention diagnostic", "",
                  "Claude's separate diagnostic measures slot-zero attention at the bootstrap forward, averaged over layers, "
                  "heads, and four seeds. The retained 28 logit-replica checks all have maximum absolute difference zero.", "",
                  "| Fragment beta | No-sink slot-zero mass | BOS slot-zero mass |", "| --- | ---: | ---: |"]
        for beta in (0., .25):
            values = []
            for mode in ("none", "bos"):
                rows = [r for r in diagnostic_rows if r["kind"] == "fragment" and r["beta"] == beta and r["sink"] == mode]
                if {r["seed"] for r in rows} != {11, 23, 47, 89} or len(rows) != 4:
                    raise ValueError("unexpected diagnostic seeds")
                values.append(sum(r["slot0_mass_mean"] for r in rows) / len(rows))
            lines.append(f"| {beta:g} | {values[0]:.4f} | {values[1]:.4f} |")
        lines += ["", "This demonstrates concentrated slot-zero attention in this measured forward. It does not establish the "
                  "absence of sinks at other positions or later steps, explain output quality, or change P1–P3. "
                  f"[Diagnostic data]({diagnostic.parent.name}/{diagnostic.name}), SHA-256 "
                  f"`{hashlib.sha256(diagnostic.read_bytes()).hexdigest()}`; "
                  f"[independent diagnostic script]({diagnostic.parent.name}/kv-dreaming-bootstrap-attention-mass-on-slot-zero-diagnostic.py)."]
    lines += ["", "## Scope and evidence", "",
              "Adding BOS also shifts fragment/bootstrap positions, protects three rather than four fragment tokens, and advances the first "
              "reduction by one sample. This compound intervention cannot isolate an attention-sink mechanism. Claude's separate slot-zero "
              "attention diagnostic is descriptive and cannot change the primary criteria. "
              "[Claude's independent audit addendum](claude-260930-0215-kv-dreaming-paired-results-audit.md) "
              "confirms P1–P3, the 1,024 exact gates, the comparison/consistency/recall hashes, and the stop verdict.", "",
              f"- [Frozen preregistration](260930-kv-dreaming-fragment-sink-preregistration.md)",
              f"- [Matched primary comparison]({prefix}/comparison.json)",
              f"- [Consistency checks and per-direction metrics]({prefix}/consistency.json)",
              f"- [M4 analysis]({prefix}/recall.json)",
              f"- [Separate event/token replay]({prefix}/fragment-crosscheck.json) and [script]({prefix}/crosscheck_fragments.py)",
              f"- [Raw manifests, tokens, text, and step logs]({prefix}/raw/)",
              f"- [Frozen plan]({prefix}/plan.json), [freeze packet]({prefix}/freeze.json), and [calibration metadata]({prefix}/calibration.json)",
              f"- [Launcher index]({prefix}/launcher-index.json)", "",
              "The checker validates recorded gates; full successful-step logits were not retained. Terminal cache tensors remain on "
              "`kv-dream-runs:/paired-4b-sink-v1/` and were not downloaded or independently inspected. Download inventories bind the local "
              "logs with SHA-256 and retain remote tensor paths/sizes. Collection and analysis ran no model inference.", "",
              "| Artifact | SHA-256 |", "| --- | --- |"]
    for name in ("plan.json", "freeze.json", "comparison.json", "consistency.json", "recall.json", "fragment-crosscheck.json"):
        lines.append(f"| {name} | `{hashlib.sha256((directory / name).read_bytes()).hexdigest()}` |")
    output.write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(output), "primary_verdict": comparison["primary_verdict"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.directory, args.output)
