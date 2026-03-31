#!/usr/bin/env python3
"""
tools/runner/run.py
Experiment runner for pla-learned-index-bench.

Usage:
  python3 tools/runner/run.py --config configs/exp_example.yaml
  python3 tools/runner/run.py --config configs/exp_example.yaml --smoke
  python3 tools/runner/run.py --config configs/exp_example.yaml --dry-run

The runner:
  1. Reads YAML config and expands the cartesian product.
  2. For each case: cmake -DPLA_ALGO=... + build + run bench binary.
  3. Appends JSONL to results/raw/<scenario>.jsonl.
  4. On completion writes results/agg/metadata.json with reproducibility info.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import yaml  # pip install pyyaml

# ── repo root (two levels up from this file) ─────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILD_DIR  = REPO_ROOT / "build"
RESULTS_RAW = REPO_ROOT / "results" / "raw"
RESULTS_AGG = REPO_ROOT / "results" / "agg"


# ── YAML loading ─────────────────────────────────────────────────────────────
def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_matrix(cfg: Dict[str, Any], smoke: bool) -> List[Dict[str, Any]]:
    """Expand config matrix into list of individual experiment cases."""
    matrix = cfg.get("matrix", {})
    keys   = list(matrix.keys())
    values = [matrix[k] for k in keys]

    cases = []
    for combo in itertools.product(*values):
        case = dict(zip(keys, combo))
        case["exp_name"] = cfg.get("exp_name", "unnamed")
        case["n_keys"]   = cfg.get("n_keys", 1_000_000)
        case["queries"]  = cfg.get("queries", 1_000_000)
        case.update(cfg.get("extra_flags", {}))
        cases.append(case)

    if smoke:
        # Smoke mode: first 4 cases only, shrink dataset size.
        cases = cases[:4]
        for c in cases:
            c["n_keys"]  = 100_000
            c["queries"] = 10_000

    return cases


# ── CMake build ───────────────────────────────────────────────────────────────
def cmake_build(pla_algo: str, extra_defs: Dict[str, str] | None = None,
                jobs: int = os.cpu_count() or 4) -> bool:
    """Configure + build for the given PLA algorithm. Returns True on success."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    defs = [
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DPLA_ALGO={pla_algo}",
        f"-DBUILD_TESTS=OFF",
    ]
    if extra_defs:
        defs += [f"-D{k}={v}" for k, v in extra_defs.items()]

    # Configure.
    cmd_cfg = ["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR)] + defs
    r = subprocess.run(cmd_cfg, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[cmake configure FAILED]\n{r.stderr}", file=sys.stderr)
        return False

    # Build.
    cmd_build = ["cmake", "--build", str(BUILD_DIR), "-j", str(jobs)]
    r = subprocess.run(cmd_build, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[cmake build FAILED]\n{r.stderr}", file=sys.stderr)
        return False

    return True


# ── Binary name per scenario ──────────────────────────────────────────────────
SCENARIO_BINARY = {
    "pla_only": "pla_build_bench",
    "inmem":    "lookup_bench",
    "dynamic":  "dynamic_bench",
    "ondisk":   "ondisk_bench",
}


def build_cmd(case: Dict[str, Any]) -> List[str]:
    """Construct the benchmark command for a given case."""
    scenario = case["scenario"]
    binary   = BUILD_DIR / SCENARIO_BINARY.get(scenario, scenario)
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    cmd = [str(binary)]
    cmd += ["--algo",    case["pla"]]
    cmd += ["--epsilon", str(case["epsilon"])]
    cmd += ["--threads", str(case["threads"])]
    cmd += ["--n",       str(case["n_keys"])]
    cmd += ["--queries", str(case.get("queries", 1_000_000))]
    cmd += ["--exp-id",  f"{case['exp_name']}_{case['pla']}_e{case['epsilon']}"]

    # Dataset argument.
    ds = case.get("dataset", "")
    if ds and not ds.startswith("synth_"):
        # Real dataset file.
        ds_path = REPO_ROOT / "data" / ds
        if ds_path.exists():
            cmd += ["--dataset", str(ds_path)]
        else:
            print(f"  [WARN] dataset not found: {ds_path}; using synthetic", file=sys.stderr)
            cmd += ["--dist", "uniform"]
    else:
        # Synthetic.
        dist = ds.split("_")[1] if "_" in ds else "uniform"
        cmd += ["--dist", dist]

    if case.get("workload"):
        cmd += ["--workload", case["workload"]]

    if scenario == "ondisk" and case.get("fetch_strategy", -1) >= 0:
        cmd += ["--fetch-strategy", str(case["fetch_strategy"])]

    return cmd


# ── Environment for a case ────────────────────────────────────────────────────
def build_env(case: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(case.get("threads", 1))
    return env


# ── numactl wrapper ───────────────────────────────────────────────────────────
def wrap_numactl(cmd: List[str], numa_node: int = 0) -> List[str]:
    if shutil.which("numactl"):
        return ["numactl", f"--cpunodebind={numa_node}",
                f"--membind={numa_node}"] + cmd
    return cmd


# ── Git metadata ──────────────────────────────────────────────────────────────
def git_rev(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def submodule_hashes(repo: Path) -> Dict[str, str]:
    r = subprocess.run(["git", "-C", str(repo), "submodule", "status"],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            out[parts[1]] = parts[0].lstrip("+-")
    return out


def lscpu_info() -> str:
    r = subprocess.run(["lscpu"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def write_metadata(cases: List[Dict[str, Any]], start_ts: str):
    meta = {
        "timestamp":        start_ts,
        "git_rev":          git_rev(REPO_ROOT),
        "submodule_hashes": submodule_hashes(REPO_ROOT),
        "platform":         platform.platform(),
        "python":           sys.version,
        "lscpu":            lscpu_info(),
        "n_cases":          len(cases),
    }
    RESULTS_AGG.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_AGG / "metadata.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[metadata] Written to {out_path}")


# ── Main runner ───────────────────────────────────────────────────────────────
def run_case(case: Dict[str, Any], dry_run: bool = False) -> bool:
    """Run one experiment case. Returns True on success."""
    scenario = case["scenario"]
    label    = (f"{case['pla']}/eps={case['epsilon']}/thr={case['threads']}/"
                f"{case['dataset']}/{case['workload']}")
    print(f"  → {scenario:10s}  {label}")

    cmd = build_cmd(case)
    env = build_env(case)
    if not dry_run:
        cmd = wrap_numactl(cmd)

    if dry_run:
        print(f"    [dry-run] {' '.join(cmd)}")
        return True

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=600)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"    [FAILED in {elapsed:.1f}s] stderr:\n{result.stderr[:500]}",
              file=sys.stderr)
        return False

    # Append each JSONL line to results/raw/<scenario>.jsonl
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_RAW / f"{scenario}.jsonl"
    with open(out_file, "a") as f:
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                f.write(line + "\n")

    print(f"    [OK  in {elapsed:.1f}s]")
    return True


def main():
    ap = argparse.ArgumentParser(description="pla-learned-index-bench runner")
    ap.add_argument("--config",  required=True, help="Path to YAML experiment config")
    ap.add_argument("--smoke",   action="store_true", help="Quick smoke run (tiny data)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument("--no-build", action="store_true", help="Skip cmake build step")
    ap.add_argument("--jobs",    type=int, default=os.cpu_count() or 4)
    ap.add_argument("--filter-scenario", help="Only run this scenario")
    ap.add_argument("--filter-pla",      help="Only run this PLA algo")
    args = ap.parse_args()

    cfg   = load_config(Path(args.config))
    cases = expand_matrix(cfg, smoke=args.smoke)

    # Apply filters.
    if args.filter_scenario:
        cases = [c for c in cases if c["scenario"] == args.filter_scenario]
    if args.filter_pla:
        cases = [c for c in cases if c["pla"] == args.filter_pla]

    print(f"Experiment: {cfg.get('exp_name','?')}  ({len(cases)} cases)")
    start_ts = datetime.datetime.utcnow().isoformat() + "Z"

    # Group cases by PLA algo to minimise rebuilds.
    from collections import defaultdict
    by_algo: Dict[str, List] = defaultdict(list)
    for c in cases:
        by_algo[c["pla"]].append(c)

    n_ok = 0; n_fail = 0
    for algo, algo_cases in by_algo.items():
        # Rebuild once per PLA algo (unless --no-build).
        if not args.no_build and not args.dry_run:
            print(f"\n── Building  PLA_ALGO={algo} ──")
            ok = cmake_build(algo, jobs=args.jobs)
            if not ok:
                print(f"[ERROR] Build failed for PLA_ALGO={algo}; skipping {len(algo_cases)} cases",
                      file=sys.stderr)
                n_fail += len(algo_cases)
                continue

        print(f"\n── Running  PLA_ALGO={algo}  ({len(algo_cases)} cases) ──")
        for case in algo_cases:
            try:
                ok = run_case(case, dry_run=args.dry_run)
            except FileNotFoundError as e:
                print(f"    [SKIP] {e}", file=sys.stderr)
                ok = False
            except subprocess.TimeoutExpired:
                print(f"    [TIMEOUT]", file=sys.stderr)
                ok = False
            if ok: n_ok += 1
            else:  n_fail += 1

    write_metadata(cases, start_ts)

    print(f"\n{'='*50}")
    print(f"Done: {n_ok} OK, {n_fail} failed")
    print(f"Results: {RESULTS_RAW}")

    # Call aggregation + plotting.
    if not args.dry_run and n_ok > 0:
        print("\nRunning aggregate + plots…")
        subprocess.run([sys.executable,
                        str(REPO_ROOT / "tools" / "viz" / "aggregate.py"),
                        "--input", str(RESULTS_RAW),
                        "--output", str(RESULTS_AGG)],
                       check=False)
        subprocess.run([sys.executable,
                        str(REPO_ROOT / "tools" / "viz" / "plots.py"),
                        "--input", str(RESULTS_AGG / "results.csv"),
                        "--output", str(RESULTS_AGG)],
                       check=False)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
