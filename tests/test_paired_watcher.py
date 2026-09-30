"""The capacity fallback must never duplicate active work or bypass stop confirmation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class EndFirstPoll(Exception):
    pass


@pytest.mark.parametrize("backlog,running,deadline,stopped,should_launch", [
    (16, 0, "2000-01-01T00:00:00+00:00", True, True),
    (16, 0, "2100-01-01T00:00:00+00:00", True, False),
    (15, 1, "2000-01-01T00:00:00+00:00", True, False),
    (16, 0, "2000-01-01T00:00:00+00:00", False, False),
])
def test_capacity_fallback_guards(tmp_path, monkeypatch, backlog, running, deadline, stopped, should_launch):
    script = Path(__file__).resolve().parents[1] / "scripts/watch_paired_artifacts.py"
    spec = importlib.util.spec_from_file_location("watch_paired_for_test", script)
    watch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watch)
    isolated_repo = tmp_path / "repo"
    (isolated_repo / "artifacts").mkdir(parents=True)
    (isolated_repo / "artifacts/paired-plan-reviewed.json").write_text(json.dumps({"paired_directions": 32}))
    monkeypatch.setattr(watch, "__file__", str(isolated_repo / "scripts/watch_paired_artifacts.py"))
    calls, launches = [], []
    monkeypatch.setattr(watch.sys, "argv", [str(script), "--output", str(tmp_path / "out"), "--app-id", "ap-test",
                                            "--allow-a100-fallback-after", deadline])

    def fake_run(argv, **kwargs):
        calls.append(argv)
        code, data = 0, {}
        if "collect_paired_artifacts.py" in argv[1]:
            code, data = 75, {"status": "no_committed_outputs_yet"}
        elif "paired_queue_status.py" in argv[1]:
            data = {"backlog": backlog, "total_tasks": running, "running_inputs": running}
        elif argv[:3] == ["modal", "app", "list"]:
            data = [{"App ID": "ap-test", "State": "stopped" if stopped else "ephemeral (detached)"}]
        elif argv[:3] != ["modal", "app", "stop"]:
            raise AssertionError(f"unexpected command: {argv}")
        return SimpleNamespace(returncode=code, stdout=json.dumps(data), stderr="")

    def fake_launch(argv, **kwargs):
        launches.append((argv, kwargs))
        return SimpleNamespace(pid=123)

    def end_poll(seconds):
        raise EndFirstPoll()

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    monkeypatch.setattr(watch.subprocess, "Popen", fake_launch)
    monkeypatch.setattr(watch.time, "sleep", end_poll)
    try:
        result = watch.main()
        assert result == 1 and not stopped
    except EndFirstPoll:
        pass
    assert bool(launches) == should_launch
    if launches:
        argv, kwargs = launches[0]
        assert len(launches) == 1 and argv[-1] == "paired-4b-v1b"
        assert kwargs["env"]["KV_DREAM_GPU"] == "A100-40GB"
        assert ["modal", "app", "stop", "ap-test"] in calls
        assert ["modal", "app", "list", "--json"] in calls
    if running or deadline.startswith("2100"):
        assert not any(call[:3] == ["modal", "app", "stop"] for call in calls)


def test_completed_index_refreshes_listing_before_analysis(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/watch_paired_artifacts.py"
    spec = importlib.util.spec_from_file_location("watch_paired_race_test", source)
    watch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watch)
    repo, output = tmp_path / "repo", tmp_path / "out"
    (repo / "artifacts").mkdir(parents=True)
    (repo / "artifacts/paired-plan-reviewed.json").write_text(json.dumps({"paired_directions": 2}))
    (repo / "modal/runs-index").mkdir(parents=True)
    (repo / "modal/runs-index/paired-4b-v1.json").write_text(json.dumps({"runs": [{"returncode": 0}]}))
    early = output / "raw/group/evict-leader/summary.json"
    early.parent.mkdir(parents=True)
    early.write_text("{}")
    monkeypatch.setattr(watch, "__file__", str(repo / "scripts/watch_paired_artifacts.py"))
    monkeypatch.setattr(watch.sys, "argv", [str(source), "--output", str(output), "--app-id", "ap-test"])
    collections = []

    def fake_run(argv, **kwargs):
        if "collect_paired_artifacts.py" in argv[1]:
            collections.append(argv)
            if len(collections) == 2:
                late = output / "raw/group/merge-leader/summary.json"
                late.parent.mkdir(parents=True)
                late.write_text("{}")
        elif "check_paired_artifacts.py" in argv[1]:
            assert len(collections) == 2
            (output / "consistency.json").write_text(json.dumps({"all_expected_artifacts_consistent": True}))
        elif "analyze-pairs" in argv:
            (output / "recall.json").write_text("{}")
        elif "report_paired_results.py" in argv[1]:
            output.with_suffix(".md").write_text("report")
        else:
            raise AssertionError(f"unexpected command: {argv}")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    assert watch.main() == 0
    status = json.loads((output / "watcher-status.json").read_text())
    assert status["state"] == "analysis_complete" and status["present_directions"] == 2
    assert (output / "launcher-index.json").exists()
