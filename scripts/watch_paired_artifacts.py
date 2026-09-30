"""Bounded read-only watcher: collect committed outputs, check, analyze, report."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tag", default="paired-4b-v1")
    p.add_argument("--max-wait-seconds", type=int, default=7200)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--app-id", required=True)
    p.add_argument("--allow-a100-fallback-after", help="Explicitly authorized UTC deadline; one fallback only")
    args = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "watcher-status.json"
    started = time.monotonic()

    def status(state, **kwargs):
        data = {"state": state, "tag": args.tag, "updated_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 1), **kwargs}
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(data), flush=True)

    def run(argv):
        return subprocess.run(argv, cwd=repo, capture_output=True, text=True, timeout=300)

    plan = repo / "artifacts/paired-plan-reviewed.json"
    expected = json.loads(plan.read_text())["paired_directions"]
    app_id = args.app_id
    fallback_done = False
    fallback_deadline = (datetime.fromisoformat(args.allow_a100_fallback_after)
                         if args.allow_a100_fallback_after else None)
    context = {"tag": args.tag, "app_id": app_id, "gpu": "L40S", "original_app_id": app_id,
               "fallback_authorization": "#lab 20629" if fallback_deadline else None,
               "fallback_not_before_utc": args.allow_a100_fallback_after,
               "plan_sha256": __import__("hashlib").sha256(plan.read_bytes()).hexdigest()}
    context_path = args.output / "execution-context.json"
    context_path.write_text(json.dumps(context, indent=2) + "\n")
    while time.monotonic() - started < args.max_wait_seconds:
        result = run([sys.executable, str(repo / "scripts/collect_paired_artifacts.py"),
                      "--tag", args.tag, "--output", str(args.output)])
        if result.returncode not in (0, 75):
            status("collection_error", stderr=result.stderr[-2000:])
            return 1
        count = len(list((args.output / "raw").glob("*/*-leader/summary.json")))
        index_path = repo / "modal/runs-index" / (args.tag + ".json")
        index_finished = index_path.exists() and len(json.loads(index_path.read_text()).get("runs", [])) == expected // 2
        if count == expected or index_finished:
            # The final launcher index may arrive after the directory listing
            # used above. Refresh after observing it before freezing analysis.
            if index_finished and count < expected:
                refreshed = run([sys.executable, str(repo / "scripts/collect_paired_artifacts.py"),
                                 "--tag", args.tag, "--output", str(args.output)])
                if refreshed.returncode not in (0, 75):
                    status("collection_error", stderr=refreshed.stderr[-2000:])
                    return 1
                count = len(list((args.output / "raw").glob("*/*-leader/summary.json")))
            status("analyzing", present_directions=count, expected_directions=expected)
            if count == 0:
                status("finished_without_direction_artifacts", index=str(index_path))
                return 1
            for argv in (
                [sys.executable, str(repo / "scripts/check_paired_artifacts.py"), "--input-root", str(args.output / "raw"),
                 "--plan", str(plan), "--output", str(args.output / "consistency.json")],
                [str(repo / ".venv/bin/kv-dream"), "analyze-pairs", "--input-root", str(args.output / "raw"),
                 "--output", str(args.output / "recall.json")],
                [sys.executable, str(repo / "scripts/report_paired_results.py"), "--directory", str(args.output),
                 "--output", str(args.output.with_suffix(".md"))],
            ):
                result = run(argv)
                if result.returncode:
                    status("analysis_error", command=argv, stderr=result.stderr[-2000:], stdout=result.stdout[-1000:])
                    return 1
            if index_finished:
                (args.output / "launcher-index.json").write_bytes(index_path.read_bytes())
            check = json.loads((args.output / "consistency.json").read_text())
            status("analysis_complete", present_directions=count, expected_directions=expected,
                   all_expected_artifacts_consistent=check["all_expected_artifacts_consistent"],
                   report=str(args.output.with_suffix(".md")))
            return 0
        collection = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else None
        queue = None
        if app_id:
            query = run([sys.executable, str(repo / "scripts/paired_queue_status.py"), "--app-id", app_id])
            if query.returncode == 0:
                queue = json.loads(query.stdout)
        if (fallback_deadline and not fallback_done and datetime.now(timezone.utc) >= fallback_deadline
                and queue and queue["backlog"] == expected // 2 and queue["total_tasks"] == 0
                and queue["running_inputs"] == 0 and count == 0 and result.returncode == 75):
            status("stopping_queued_original_for_authorized_fallback", queue=queue)
            stopped = run(["modal", "app", "stop", app_id])
            listing = run(["modal", "app", "list", "--json"])
            app = next((a for a in json.loads(listing.stdout) if a["App ID"] == app_id), None) if listing.returncode == 0 else None
            if stopped.returncode or app is None or app["State"] != "stopped":
                status("fallback_stop_not_confirmed", stop_stdout=stopped.stdout, stop_stderr=stopped.stderr)
                return 1
            context["original_stop_confirmed_utc"] = datetime.now(timezone.utc).isoformat()
            context["original_queue_before_stop"] = queue
            args.tag = "paired-4b-v1b"
            context.update(tag=args.tag, gpu="A100-40GB", app_id=None)
            launch_log = args.output / "fallback-launcher.log"
            with launch_log.open("x") as log:
                launched = subprocess.Popen(["modal", "run", "--detach", "modal/modal-kv-dream-launcher.py::pairs",
                    "--plan", str(plan), "--calibration", "/runs/calib/qwen3-4b-base-cuda-v2.pt", "--tag", args.tag],
                    cwd=repo, env={**os.environ, "KV_DREAM_GPU": "A100-40GB"}, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            context["fallback_launcher_pid"] = launched.pid
            context["fallback_launched_utc"] = datetime.now(timezone.utc).isoformat()
            context_path.write_text(json.dumps(context, indent=2) + "\n")
            fallback_done, app_id = True, None
            status("authorized_fallback_launched", launcher_pid=launched.pid, gpu="A100-40GB", previous_queue=queue)
        else:
            status("waiting_for_committed_outputs", present_directions=count, expected_directions=expected,
                   queue=queue, collection=collection)
        if fallback_done and app_id is None:
            ids = re.findall(r"ap-[A-Za-z0-9]+", (args.output / "fallback-launcher.log").read_text())
            if ids:
                app_id = ids[0]
                context["app_id"] = app_id
                context_path.write_text(json.dumps(context, indent=2) + "\n")
        time.sleep(args.poll_seconds)
    status("wait_limit_reached", max_wait_seconds=args.max_wait_seconds)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
