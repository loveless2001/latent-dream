"""Modal launcher for kv-dreaming (owned by claude; the harness itself is codex's).

Runs the unmodified `kv_dreaming` CLI on a GPU. Only `src/` and `data/` are
mounted, so harness fixes flow in without a fork. Outputs land on the
`kv-dream-runs` volume; fetch with `modal volume get kv-dream-runs <tag> runs/<tag>`.

Usage (from the project root):
  modal run modal/modal-kv-dream-launcher.py::prefetch --model 4b
  modal run modal/modal-kv-dream-launcher.py::cli --args "calibrate --output /runs/calib/4b.pt" --model 4b
  modal run --detach modal/modal-kv-dream-launcher.py::matrix --plan artifacts/initial-matrix.json \
      --model 4b --calibration /runs/calib/4b.pt --tag matrix-2k-4b
  modal run modal/modal-kv-dream-launcher.py::script --path <local.py> --model 4b --args "<argv>"

Determinism: fp32 weights, TF32 disabled (NVIDIA_TF32_OVERRIDE=0), deterministic
cuBLAS workspace, one run per container, no automatic retries.
"""

import json
import os
import shlex
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = os.environ.get("KV_DREAM_GPU", "L40S")  # 48 GB: fits 4B and 8B in fp32
MODELS = {  # pinned base checkpoints; same Qwen3 attention layout and token IDs
    "0.6b": ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"),
    "4b": ("Qwen/Qwen3-4B-Base", "906bfd4b4dc7f14ee4320094d8b41684abff8539"),
    "8b": ("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"),
}

app = modal.App("kv-dream")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.7.1", "transformers==4.55.2", "safetensors==0.5.3")
         .env({"PYTHONPATH": "/root/kv/src", "NVIDIA_TF32_OVERRIDE": "0",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false",
               "HF_HOME": "/hf"})
         .add_local_dir(ROOT / "src", "/root/kv/src")
         .add_local_dir(ROOT / "data", "/root/kv/data"))
runs = modal.Volume.from_name("kv-dream-runs", create_if_missing=True)
hf = modal.Volume.from_name("kv-dream-hf-cache", create_if_missing=True)
VOLS = {"/runs": runs, "/hf": hf}


def OFFLINE_ENV() -> dict:
    """Runs use only the prefetched cache; no Hub lookups (prefetch stays online)."""
    return {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def model_flags(model: str) -> list[str]:
    name, rev = MODELS[model]
    return ["--model", name, "--revision", rev, "--device", "cuda", "--local-files-only"]


def with_model(argv: list[str], model: str) -> list[str]:
    """Insert pinned model/device flags after the CLI subcommand unless given."""
    if argv[0] == "plan-matrix" or "--model" in argv:
        return argv
    return argv[:1] + model_flags(model) + argv[1:]


@app.function(image=image, volumes={"/hf": hf}, timeout=3600, retries=0)
def prefetch_remote(model: str):
    from huggingface_hub import snapshot_download
    name, rev = MODELS[model]
    path = snapshot_download(name, revision=rev)
    hf.commit()
    return path


@app.function(image=image, gpu=GPU, volumes=VOLS, timeout=4 * 3600, retries=0,
              max_containers=40, scaledown_window=2)
def run_cli(argv: list[str]) -> dict:
    """Run one `kv-dream` CLI invocation; never raises so a matrix keeps going."""
    import subprocess
    runs.reload()
    proc = subprocess.run(["python", "-m", "kv_dreaming.cli", *argv], cwd="/root/kv",
                          capture_output=True, text=True, env=OFFLINE_ENV())
    runs.commit()
    return {"argv": argv, "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


@app.function(image=image, gpu=GPU, volumes=VOLS, timeout=2 * 3600, retries=0)
def run_script(source: str, argv: list[str]) -> dict:
    """Execute an audit/diagnostic script (sent as text) against the mounted harness."""
    import subprocess
    Path("/root/kv/remote_script.py").write_text(source)
    runs.reload()
    proc = subprocess.run(["python", "remote_script.py", *argv], cwd="/root/kv",
                          capture_output=True, text=True, env=OFFLINE_ENV())
    runs.commit()
    return {"returncode": proc.returncode, "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-8000:]}


@app.local_entrypoint()
def prefetch(model: str = "4b"):
    print(prefetch_remote.remote(model))


@app.local_entrypoint()
def cli(args: str, model: str = "4b"):
    result = run_cli.remote(with_model(shlex.split(args), model))
    print(json.dumps(result, indent=2))
    if result["returncode"]:
        raise SystemExit(result["returncode"])


@app.local_entrypoint()
def script(path: str, model: str = "4b", args: str = ""):
    name, rev = MODELS[model]
    argv = shlex.split(args) + ["--model", name, "--revision", rev]
    result = run_script.remote(Path(path).read_text(), argv)
    print(result["stdout"])
    print(result["stderr"][-3000:])
    if result["returncode"]:
        raise SystemExit(result["returncode"])


@app.local_entrypoint()
def matrix(plan: str, calibration: str, tag: str, model: str = "4b", snapshot_every: int = -1):
    """Fan out every row of a `kv-dream plan-matrix` file, one container per run.

    snapshot_every >= 0 overrides the plan's periodic-snapshot cadence (storage
    only; generation is unaffected). The override is recorded in each run's
    manifest config and in the index.
    """
    rows = json.loads(Path(plan).read_text())["runs"]
    if snapshot_every >= 0:
        rows = [{**cfg, "snapshot_every": snapshot_every} for cfg in rows]
    jobs = []
    for cfg in rows:
        name = (f"{cfg['initial_state']}-{cfg['policy']}-i{cfg['initialization_seed']}"
                f"-s{cfg['sampling_seed']}")
        if (cfg.get("alpha_k", 1.0), cfg.get("alpha_v", 1.0)) != (1.0, 1.0):
            name += f"-ak{cfg['alpha_k']:g}-av{cfg['alpha_v']:g}"  # keep sweep names unique
        if cfg.get("initial_state") == "fragment":
            name += f"-b{cfg.get('fragment_beta', 0.0):g}-{cfg.get('fragment_sink', 'none')}"
        argv = ["run", "--calibration", calibration, "--output", f"/runs/{tag}/{name}"]
        for key, value in cfg.items():
            argv += ["--" + key.replace("_", "-"), str(value)]
        jobs.append(with_model(argv, model))
    results = list(run_cli.map(jobs, return_exceptions=True))
    index = [r if isinstance(r, dict) else {"error": repr(r)} for r in results]
    out = Path(__file__).resolve().parent / "runs-index" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"plan": plan, "calibration": calibration, "model": MODELS[model],
                               "snapshot_every_override": snapshot_every, "runs": index},
                              indent=2) + "\n")
    failed = sum(1 for r in index if r.get("returncode") != 0)
    print(f"{len(index)} runs, {failed} failed; index -> {out}")


@app.local_entrypoint()
def pairs(plan: str, calibration: str, tag: str, only: int = -1):
    """One `kv-dream paired` group per container (both leader directions share one
    loaded runtime and one initial cache). Rows carry their own `preset`; only the
    CUDA device and offline flags are added. `only` runs a single row (audit)."""
    rows = json.loads(Path(plan).read_text())["runs"]
    if only >= 0:
        rows = [rows[only]]
    jobs = []
    for cfg in rows:
        name = f"{cfg['initial_state']}-i{cfg['initialization_seed']}-s{cfg['sampling_seed']}"
        if cfg["initial_state"] == "random":
            name += f"-a{cfg['alpha_k']:g}"
        if cfg["initial_state"] == "fragment":
            name += f"-b{cfg['fragment_beta']:g}"
        argv = ["paired", "--calibration", calibration, "--output", f"/runs/{tag}/{name}",
                "--device", "cuda", "--local-files-only"]
        for key, value in cfg.items():
            argv += ["--" + key.replace("_", "-"), str(value)]
        jobs.append(argv)
    results = list(run_cli.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    index = [r if isinstance(r, dict) else {"error": repr(r)} for r in results]
    out = Path(__file__).resolve().parent / "runs-index" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"plan": plan, "calibration": calibration, "only": only,
                               "runs": index}, indent=2) + "\n")
    failed = sum(1 for r in index if r.get("returncode") != 0)
    print(f"{len(index)} groups, {failed} failed; index -> {out}")
