import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_collector_waits_for_terminal_summary_and_refuses_mutated_terminal_file(tmp_path, monkeypatch):
    files = {"batch/group/group.json": b"{}", "batch/group/evict-leader/steps.jsonl": b"partial\n"}
    reads = []

    class Volume:
        @staticmethod
        def from_name(name):
            return Volume()

        def iterdir(self, path, **kwargs):
            return [SimpleNamespace(path=name, type=1, size=len(data), mtime=1) for name, data in files.items()]

        def read_file(self, name):
            reads.append(name)
            yield files[name]

    import sys
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=Volume, exception=SimpleNamespace(NotFoundError=FileNotFoundError)))
    monkeypatch.setitem(sys.modules, "modal.volume", SimpleNamespace(FileEntryType=SimpleNamespace(FILE=1)))
    script = Path(__file__).resolve().parents[1] / "scripts/collect_paired_artifacts.py"
    spec = importlib.util.spec_from_file_location("paired_collector_for_test", script)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    monkeypatch.setattr(sys, "argv", [str(script), "--tag", "batch", "--output", str(tmp_path)])
    collector.main()
    assert reads == ["batch/group/group.json"]
    assert not (tmp_path / "raw/group/evict-leader/steps.jsonl").exists()
    files["batch/group/evict-leader/steps.jsonl"] = b"complete\ncomplete\n"
    files["batch/group/evict-leader/summary.json"] = b"{}"
    files["batch/group/evict-leader/final-state.pt"] = b"tensor"
    collector.main()
    target = tmp_path / "raw/group/evict-leader/steps.jsonl"
    assert target.read_bytes() == b"complete\ncomplete\n"
    assert not (target.parent / "final-state.pt").exists()
    files["batch/group/evict-leader/steps.jsonl"] += b"unexpected change"
    with pytest.raises(RuntimeError, match="changed size"):
        collector.main()
    assert target.read_bytes() == b"complete\ncomplete\n"
