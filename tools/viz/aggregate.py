#!/usr/bin/env python3
"""
tools/viz/aggregate.py
Aggregate raw JSONL results into a single CSV for plotting.

Usage:
  python3 tools/viz/aggregate.py --input results/raw --output results/agg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


# ── JSONL loading ─────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{lineno}: {e}", file=sys.stderr)
    return rows


# ── Derived metrics ───────────────────────────────────────────────────────────
def add_derived(row: Dict[str, Any]) -> Dict[str, Any]:
    instr  = row.get("instructions", 0) or 1
    cm     = row.get("cache_misses", 0) or 0
    cycles = row.get("cycles", 0) or 0
    row["cache_miss_rate"] = cm / instr if instr else 0.0
    row["ipc"]             = instr / cycles if cycles else 0.0
    return row


# ── Percentile utilities ──────────────────────────────────────────────────────
def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = int(p / 100.0 * (len(sv) - 1))
    return sv[min(idx, len(sv)-1)]


# ── CSV output columns ────────────────────────────────────────────────────────
COLUMNS = [
    "exp_id", "scenario", "index", "pla", "epsilon", "threads",
    "dataset", "workload",
    "build_ms", "seg_cnt", "bytes_index",
    "ops_s", "p50_ns", "p95_ns", "p99_ns",
    "cache_misses", "branches", "branch_misses", "instructions", "cycles",
    "rss_mb", "fetch_strategy", "io_pages",
    "cache_miss_rate", "ipc",
    "max_err", "retrain_ms", "retrain_count", "n_keys", "dup_runs",
    # IM-A segment statistics
    "seg_len_mean", "seg_len_p50", "seg_len_p95",
    "rank_span_mean", "slope_mean", "slope_std", "intercept_std",
    # IM-B routing metadata
    "routing", "index_levels", "seg_cnt_l1",
    # DW-A/B retrain analysis
    "retrain_p50_ms", "retrain_p95_ms", "retrain_window_p99_ns",
    # OD-A..F page-level metrics
    "granularity", "page_align", "target_rp",
    "io_pages_mean", "io_pages_p50", "io_pages_p95", "io_pages_p99",
]


def main():
    ap = argparse.ArgumentParser(description="Aggregate JSONL → CSV")
    ap.add_argument("--input",  required=True, help="Directory containing *.jsonl files")
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for jsonl_file in sorted(in_dir.glob("*.jsonl")):
        rows = load_jsonl(jsonl_file)
        print(f"  {jsonl_file.name}: {len(rows)} rows")
        for r in rows:
            add_derived(r)
            all_rows.append(r)

    if not all_rows:
        print("[WARN] No rows found in input directory.", file=sys.stderr)
        return

    # Write CSV.
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            # Fill missing keys with empty string.
            filled = {c: row.get(c, "") for c in COLUMNS}
            writer.writerow(filled)

    print(f"[aggregate] Written {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
