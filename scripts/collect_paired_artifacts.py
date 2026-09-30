"""Download committed paired logs through read-only Modal Volume calls.

Run with the Python environment that provides Modal. Terminal cache tensors stay
on the volume; their remote paths and sizes are retained in each inventory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath

import modal
from modal.volume import FileEntryType


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="paired-4b-v1")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    volume = modal.Volume.from_name("kv-dream-runs")
    try:
        entries = list(volume.iterdir("/" + args.tag, recursive=True))
    except modal.exception.NotFoundError:
        print(json.dumps({"tag": args.tag, "status": "no_committed_outputs_yet"}))
        raise SystemExit(75)
    files = [entry for entry in entries if entry.type == FileEntryType.FILE]
    terminal_directions = {str(PurePosixPath(entry.path.lstrip("/")).parent)
                           for entry in files if PurePosixPath(entry.path).name == "summary.json"
                           and PurePosixPath(entry.path).parent.name.endswith("-leader")}
    if not files:
        raise SystemExit("No committed files yet")
    args.output.mkdir(parents=True, exist_ok=True)

    def download(entry):
        remote = PurePosixPath(entry.path.lstrip("/"))
        relative = remote.relative_to(args.tag)
        if ".." in relative.parts:
            raise ValueError("unsafe remote path")
        terminal = not remote.parent.name.endswith("-leader") or str(remote.parent) in terminal_directions
        row = {"remote_path": str(remote), "size": entry.size, "mtime": entry.mtime,
               "downloaded": terminal and remote.name != "final-state.pt"}
        if not terminal:
            row["not_downloaded_reason"] = "direction_not_terminal_yet"
        elif remote.name == "final-state.pt":
            row["not_downloaded_reason"] = "terminal_tensor_retained_on_volume"
        if row["downloaded"]:
            target = args.output / "raw" / str(relative)
            if target.exists():
                raw = target.read_bytes()
                if len(raw) != entry.size:
                    raise RuntimeError(f"previously downloaded artifact changed size: {remote}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = b"".join(volume.read_file(str(remote)))
                if len(raw) != entry.size:
                    raise RuntimeError(f"download size mismatch: {remote}")
                with target.open("xb") as f:
                    f.write(raw)
            row["sha256"] = hashlib.sha256(raw).hexdigest()
            row["local_path"] = str(target)
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        inventory = sorted(pool.map(download, files), key=lambda row: row["remote_path"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = {"volume": "kv-dream-runs", "tag": args.tag,
              "retrieved_at_utc": stamp, "files": inventory,
              "note": "Read-only retrieval. Terminal cache tensors retained on volume, not downloaded."}
    path = args.output / f"inventory-{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    summaries = list((args.output / "raw").glob("*/*-leader/summary.json"))
    print(json.dumps({"inventory": str(path), "remote_files": len(files),
                      "downloaded_files": sum(r["downloaded"] for r in inventory),
                      "direction_summaries": len(summaries)}))


if __name__ == "__main__":
    main()
