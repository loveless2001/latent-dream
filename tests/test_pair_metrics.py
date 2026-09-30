import json

from kv_dreaming.pair_metrics import FragmentTracker, recall_after, analyze_pairs, fragment_pre_reduction


def test_recall_excludes_boundary_crossing_and_preserves_null():
    tokens = [9, 10, 11, 12, 13, 14]
    result = recall_after(tokens, [[10, 11, 12], [11, 12, 13]], 1)
    assert result["recall"] == .5 and result["last_recurrence_step"] == 4
    assert recall_after(tokens, [[10, 11, 12]], None)["recall"] is None
    assert recall_after(tokens, [], 0)["recall"] is None
    assert recall_after(tokens, [[10, 11, 12]], 4)["recall"] is None


def test_fragment_boundaries_retain_pinned_originals():
    metadata = {"fragments": [{"id": 0, "source_id": "a", "virtual_positions": list(range(6)),
                               "protected_positions": [0, 1, 2, 3], "eligible_trigrams": [[1, 2, 3]]}]}
    tracker = FragmentTracker(metadata, 6)
    tracker.append()
    tracker.reduce([{"kind": "merge", "index": 4}], 10)
    b = tracker.boundaries["0"]
    assert b["first_affected_step"] == b["all_unprotected_affected_step"] == 10
    assert b["all_original_affected_step"] is None
    assert len(tracker.origins) == 6
    tracker.reduce([{"kind": "evict", "index": 4}], 11)
    assert b["first_affected_step"] == 10


def test_shifted_fragment_boundaries_do_not_treat_sink_as_fragment():
    metadata = {"fragments": [{"id": 0, "source_id": "a", "virtual_positions": list(range(1, 17)),
                               "protected_positions": [1, 2, 3], "eligible_trigrams": [[1, 2, 3]]}]}
    tracker = FragmentTracker(metadata, 65)
    for step in range(63, 76):
        tracker.append()
        tracker.reduce([{"kind": "evict", "index": 4}], step)
    boundary = tracker.boundaries["0"]
    assert boundary["first_affected_step"] == 63
    assert boundary["all_unprotected_affected_step"] == 75
    assert boundary["all_original_affected_step"] is None
    assert tracker.origins[:4] == [{0}, {1}, {2}, {3}]


def test_fragment_p3_includes_trigger_sample_and_excludes_crossing_trigram():
    metadata = {"fragments": [{"id": 0, "eligible_trigrams": [[1, 2, 3], [2, 3, 4]]}]}
    result = fragment_pre_reduction([9, 1, 2, 3, 4], metadata, 3)
    assert result["window_tokens"] == 4 and result["last_included_step"] == 3
    assert result["direction_any_hit"] and result["fragments"][0]["hit_trigrams"] == [[1, 2, 3]]
    early_eos = fragment_pre_reduction([1, 2, 3], metadata, None)
    assert early_eos["direction_any_hit"] and early_eos["window_tokens"] == 3
    short = fragment_pre_reduction([1], metadata, None)
    assert not short["trigram_opportunity"] and not short["direction_any_hit"]
    assert fragment_pre_reduction([1, 2, 3], None, None) is None


def test_cross_run_baseline_and_contamination_exclusion(tmp_path):
    for seed, source_tokens, output_tokens in (
            (11, [1, 2, 3], [99, 1, 2, 3, 4]),
            (23, [5, 6, 7], [99, 1, 2, 3, 4]),
            (47, [7, 8, 9], [99, 10, 11, 12, 13]),
            (89, [1, 2, 3], [99, 1, 2, 3, 4]),
            (91, [20, 21, 22], [99, 1]),
            (93, [5, 6, 7], [99, 1, 2, 3, 4])):
        path = tmp_path / str(seed)
        path.mkdir()
        metadata = {"kind": "fragment", "manifest_sha256": "fixed", "fragments": [
            {"id": 0, "token_ids": source_tokens, "eligible_trigrams": [source_tokens]}]}
        manifest = {"kind": "paired_shadow", "identity": {}, "source_hashes": {},
                    "calibration_sha256": "fixed", "initialization_metadata": metadata,
                    "config": {"policy": "evict", "fragment_beta": 0, "budget": 128,
                               "recent": 64, "max_tokens": 2048, "sampling_seed": 101,
                               "fragment_sink": "bos" if seed == 93 else "none",
                               "prefix_length": 65 if seed == 93 else 64,
                               "initialization_seed": seed}}
        summary = {"stop_reason": "token_limit", "fragment_recall": [{"boundaries": {
            "first_affected_step": recall_after(output_tokens, [source_tokens], 0)}}]}
        for name, value in (("manifest", manifest), ("summary", summary), ("tokens", output_tokens)):
            (path / f"{name}.json").write_text(json.dumps(value))
    report = analyze_pairs(tmp_path)
    row = next(r for r in report["fragment_comparisons"] if r["run"] == str(tmp_path / "11"))
    assert row["eligible_comparators"] == 2
    assert row["baseline_mean_recall"] == .5 and row["recall_minus_baseline"] == .5
    reasons = {r["reason"] for r in row["excluded_comparators"]}
    assert "target_trigram_present_in_comparator_seed" in reasons
    assert "no_post_boundary_trigram_opportunity" in reasons
    assert "condition_mismatch" in reasons
