#!/usr/bin/env python3
"""
tools/viz/plots.py
Generate benchmark plots from aggregated CSV.

Charts produced:
  1. epsilon vs seg_cnt       (line per pla × dataset, facet by scenario)
  2. epsilon vs build_ms      (line per pla × dataset, facet by scenario)
  3. threads vs throughput    (line per pla × dataset, facet by scenario)
  4. latency CDF              (canonical slice per scenario)
  5. cache-miss rate          (bar: pla × epsilon, aggregated)
  6. fetch_strategy vs p99/ops_s  (ondisk, canonical slice)
  7. routing comparison       (inmem, canonical dataset)
  8. seg len distribution     (line per pla × dataset)
  9. parallel build overhead  (aggregated per threads)
 10. retrain impact           (grouped by workload)
 11. ondisk Rp vs epsilon     (canonical slice)
 12. page-align benefit       (canonical slice)
 13. RSS memory               (ε vs rss_mb, 2×2 facet by scenario)
 14. index size               (ε vs bytes_index, 2×2 facet by scenario)
 15. G4 compress effect       (ondisk: compress on/off vs ops_s + bytes)
 16. dynamic throughput       (ε vs ops_s by workload, DW-B)
 17. slope / intercept std    (ε vs slope_std / intercept_std, IM-A)
 18. microarch IPC + branches (ε vs ipc / branch_miss_rate, IM-B)
 19. granularity item vs page (ondisk: ε vs io_pages / ops_s, OD-D)
 20. fetch × threads          (ondisk: fetch_strategy vs ops_s faceted by threads, OD-C)
 21. target Rp iso-Rp         (target_rp vs ε / seg_cnt, OD-A)
 22. hybrid workloads         (ondisk: ops_s / io_pages by workload, OD-F)
 23. retrain count            (ε vs retrain_count, DW-A)
 24. PLA build scaling        (threads vs build_ms for pla_only, IM-A)
 25. inmem space-time tradeoff (seg_cnt vs ops_s/p99, per routing)
 26. inmem perf counters       (ε vs cache_miss_rate/IPC/branch_miss, per routing)
 27. inmem perf scatter        (cache_miss vs ops_s, branch_miss vs p99)
 28. inmem instr-per-lookup    (seg_cnt vs instructions/lookup)

Usage:
  python3 tools/viz/plots.py --input results/agg/results.csv --output results/agg
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available; skipping plots.", file=sys.stderr)


# ── Data loading ──────────────────────────────────────────────────────────────

NUMERIC_COLS = {
    "epsilon", "threads", "seg_cnt", "bytes_index", "fetch_strategy",
    "build_ms", "ops_s", "p50_ns", "p95_ns", "p99_ns",
    "cache_misses", "instructions", "cycles", "cache_miss_rate", "ipc",
    "branches", "branch_misses", "rss_mb", "io_pages",
    "max_err", "retrain_ms", "retrain_count", "n_keys", "dup_runs",
    "n_insert", "n_lookup", "lookup_ops_s",
    "seg_len_mean", "seg_len_p50", "seg_len_p95",
    "rank_span_mean", "slope_mean", "slope_std", "intercept_std",
    "index_levels", "seg_cnt_l1",
    "retrain_p50_ms", "retrain_p95_ms", "retrain_window_p99_ns",
    "target_rp",
    "io_pages_mean", "io_pages_p50", "io_pages_p95", "io_pages_p99",
    "bytes_compressed",
}

def load_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in NUMERIC_COLS:
                v = row.get(k, "")
                try:
                    row[k] = float(v) if v != "" else 0.0
                except (ValueError, TypeError):
                    row[k] = 0.0
            rows.append(row)
    return rows


# ── Colour palette ────────────────────────────────────────────────────────────
PLA_COLORS = {"optimal": "#1f77b4", "swing": "#ff7f0e", "greedy": "#2ca02c"}

def pla_color(pla: str) -> str:
    return PLA_COLORS.get(pla, "#999999")


# ── Helpers ───────────────────────────────────────────────────────────────────

def group_by(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    g: Dict[str, List] = defaultdict(list)
    for r in rows:
        g[r.get(key, "")].append(r)
    return dict(g)


# Set by main() after loading data — controls whether dataset names appear in legends
_SINGLE_DATASET = False


def _norm_dataset(r: Dict) -> str:
    """Extract a short, stable dataset stem regardless of path prefix."""
    ds = str(r.get("dataset", ""))
    ds = ds.split("/")[-1] if "/" in ds else ds
    # Strip known prefixes that vary by clone location
    for pfx in ("sosd_ondisk_", "sosd_", "synth_"):
        if ds.startswith(pfx):
            ds = ds[len(pfx):]
    # Strip known verbose suffixes
    for suffix in ("_uint64", "_uint32", "_sorted", "_200M", "_1M", "_100K"):
        if ds.endswith(suffix):
            ds = ds[:-len(suffix)]
    return ds or "unknown"


def dataset_label(r: Dict) -> str:
    """Short readable dataset label.

    When all rows share the same dataset the label is empty so legends show
    only the PLA name.  Otherwise it includes a short dataset stem + key count.
    """
    if _SINGLE_DATASET:
        return ""

    ds = _norm_dataset(r)
    nk = r.get("n_keys", 0) or 0
    if nk >= 1_000_000_000:
        nkl = f"{nk/1_000_000_000:.0f}B"
    elif nk >= 1_000_000:
        nkl = f"{nk/1_000_000:.0f}M"
    elif nk >= 1_000:
        nkl = f"{nk/1_000:.0f}K"
    elif nk > 0:
        nkl = str(int(nk))
    else:
        nkl = ""
    return f"{ds}_{nkl}" if nkl else ds


# Markers used to distinguish datasets visually
DATASET_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]

def dataset_marker(idx: int) -> str:
    return DATASET_MARKERS[idx % len(DATASET_MARKERS)]


def _plot_grouped_lines(ax, rows, x_key, y_key, *, group_keys,
                        filter_fn=None, marker_fn=None, color_fn=None):
    """Generic: plot y vs x, one line per unique combination of group_keys.

    Duplicate x-values within a group are averaged so lines stay smooth when
    multiple workloads / routings / configs share the same nominal x (e.g. same
    epsilon but different workloads).
    """
    if filter_fn:
        rows = [r for r in rows if filter_fn(r)]

    # Build composite groups
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        parts = [str(r.get(k, "")) for k in group_keys]
        label = " / ".join(p for p in parts if p)
        if not label:
            label = str(r.get("pla", ""))
        groups[label].append(r)

    seen_any = False
    for idx, (label, prows) in enumerate(sorted(groups.items())):
        # Average y-values at each x to avoid jagged vertical scatter
        x_buckets: Dict[float, List[float]] = defaultdict(list)
        for r in prows:
            if r[y_key] > 0:
                x_buckets[float(r[x_key])].append(r[y_key])
        pts = sorted((x, sum(vals) / len(vals)) for x, vals in x_buckets.items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        pla = prows[0].get("pla", "")
        kwargs = {
            "color": color_fn(pla) if color_fn else pla_color(pla),
            "marker": marker_fn(idx) if marker_fn else dataset_marker(idx),
            "label": label,
            "markersize": 5,
        }
        ax.plot(xs, ys, **kwargs)
        seen_any = True
    return seen_any


def _hide_unused_subplots(axes, used_count: int):
    """Hide subplot axes beyond *used_count* so empty panels don't waste space."""
    flat = axes.flat if hasattr(axes, "flat") else [axes]
    for i, ax in enumerate(flat):
        if i >= used_count:
            ax.set_visible(False)


# ── Plot 1: ε vs seg_cnt ─────────────────────────────────────────────────────
def plot_epsilon_seg_cnt(rows: List[Dict], out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, scenario in zip(axes, ["pla_only", "inmem"]):
        sr = [r for r in rows if r["scenario"] == scenario]
        if not sr:
            ax.set_title(f"{scenario} (no data)")
            continue
        seen = _plot_grouped_lines(ax, sr, "epsilon", "seg_cnt",
                                   group_keys=["pla", dataset_label],
                                   filter_fn=lambda r: r["seg_cnt"] > 0)
        ax.set_title(f"{scenario}: ε vs seg_cnt")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel("Number of segments")
        ax.set_yscale("log")
        if seen:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "epsilon_vs_seg_cnt.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 2: ε vs build_ms ────────────────────────────────────────────────────
def plot_epsilon_build_ms(rows: List[Dict], out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, scenario in zip(axes, ["pla_only", "inmem"]):
        sr = [r for r in rows if r["scenario"] == scenario]
        if not sr:
            ax.set_title(f"{scenario} (no data)")
            continue
        seen = _plot_grouped_lines(ax, sr, "epsilon", "build_ms",
                                   group_keys=["pla", dataset_label],
                                   filter_fn=lambda r: r["build_ms"] > 0)
        ax.set_title(f"{scenario}: ε vs Build Time")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel("build_ms")
        ax.set_yscale("log")
        if seen:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "epsilon_vs_build_ms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 3: threads vs throughput ────────────────────────────────────────────
def plot_threads_throughput(rows: List[Dict], out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, scenario in zip(axes, ["inmem", "dynamic"]):
        sr = [r for r in rows if r["scenario"] == scenario and r["ops_s"] > 0]
        if not sr:
            ax.set_title(f"{scenario} (no data)")
            continue
        seen = _plot_grouped_lines(ax, sr, "threads", "ops_s",
                                   group_keys=["pla", dataset_label])
        ax.set_title(f"{scenario}: threads vs ops/s")
        ax.set_xlabel("Threads")
        ax.set_ylabel("ops/s")
        if seen:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "threads_vs_throughput.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 4: Latency CDF ──────────────────────────────────────────────────────
def plot_latency_cdf(rows: List[Dict], out_dir: Path):
    """Pick a canonical slice: uniform_100K for synth scenarios, sosd for ondisk."""
    fig, ax = plt.subplots(figsize=(10, 6))
    seen_any = False

    # Canonical filter: one dataset per scenario to keep the chart readable.
    canonical = {
        "pla_only": ("uniform_synth", 100000),
        "inmem":    ("synth_uniform_100000", 0),
        "dynamic":  ("synth_uniform_100000", 0),
        "ondisk":   ("sosd_ondisk_fb_200M_uint64", 0),
    }

    for scenario, (ds_hint, nk_hint) in canonical.items():
        sr = [r for r in rows if r["scenario"] == scenario]
        # Match dataset loosely
        if nk_hint > 0:
            sr = [r for r in sr if r.get("n_keys", 0) == nk_hint]
        sr = [r for r in sr if ds_hint in str(r.get("dataset", ""))]
        if not sr:
            sr = [r for r in rows if r["scenario"] == scenario]  # fallback
        # Further narrow: single-thread, canonical routing/workload
        sr = [r for r in sr if r.get("threads", 1) == 1.0]
        if scenario == "inmem":
            sr = [r for r in sr if r.get("routing", "") == "fiting-tree"]
        if scenario == "dynamic":
            sr = [r for r in sr if r.get("workload", "") == "readonly_ir0"]
        if scenario == "ondisk":
            sr = [r for r in sr if r.get("granularity", "item") == "item"
                  and str(r.get("page_align", "")).lower() in ("false", "0", "")]

        for r in sr:
            if r["p50_ns"] == 0:
                continue
            pts = [(r["p50_ns"], 50), (r["p95_ns"], 95), (r["p99_ns"], 99)]
            xs, ys = zip(*sorted(pts))
            pla = r.get("pla", "")
            lbl = f"{scenario}/{pla}/ε={int(r['epsilon'])}"
            ax.plot(xs, ys, marker="o", label=lbl, color=pla_color(pla),
                    alpha=0.7, linewidth=1.2, markersize=4)
            seen_any = True

    if not seen_any:
        ax.text(0.5, 0.5, "No latency data", ha="center", transform=ax.transAxes)
    ax.set_title("Latency CDF — canonical slice (p50/p95/p99)")
    ax.set_xlabel("Latency (ns)")
    ax.set_ylabel("Percentile")
    ax.grid(True, alpha=0.3)
    if seen_any:
        ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0, ncol=1)
        fig.subplots_adjust(right=0.68)
    else:
        fig.tight_layout()
    out = out_dir / "latency_cdf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 5: Cache-miss rate ──────────────────────────────────────────────────
def plot_cache_miss_rate(rows: List[Dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epsilons = sorted(set(int(r["epsilon"]) for r in rows if r["cache_miss_rate"] > 0))
    plas = list(PLA_COLORS.keys())
    if not epsilons:
        ax.text(0.5, 0.5, "No perf data (run with perf_stat.sh)", ha="center",
                transform=ax.transAxes)
    else:
        x = range(len(epsilons))
        width = 0.25
        for i, pla in enumerate(plas):
            vals = []
            for eps in epsilons:
                matched = [r["cache_miss_rate"] for r in rows
                           if r["pla"] == pla and int(r["epsilon"]) == eps
                           and r["cache_miss_rate"] > 0]
                vals.append(sum(matched) / len(matched) if matched else 0)
            ax.bar([xi + i * width for xi in x], vals, width, label=pla,
                   color=pla_color(pla), alpha=0.8)
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.set_xlabel("ε")
        ax.set_ylabel("Cache-miss rate (misses/instruction)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Cache-Miss Rate by ε and PLA (aggregated)")
    plt.tight_layout()
    out = out_dir / "cache_miss_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 6: fetch_strategy vs p99 / ops_s (ondisk) ──────────────────────────
def plot_ondisk_fetch(rows: List[Dict], out_dir: Path):
    """Canonical ondisk slice: item granularity, no page-align, readonly."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk"
              and r.get("granularity", "item") == "item"
              and str(r.get("page_align", "")).lower() in ("false", "0", "")
              and r.get("target_rp", 0) == 0.0]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, label in zip(axes,
                                  ["p99_ns", "ops_s"],
                                  ["p99 latency (ns)", "Throughput (ops/s)"]):
        # Group by (pla, epsilon) so each line = one PLA at one ε across fetch strategies
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in ondisk:
            groups[(r["pla"], int(r["epsilon"]))].append(r)
        seen_any = False
        for (pla, eps), prows in sorted(groups.items()):
            pts = sorted((r["fetch_strategy"], r[metric]) for r in prows if r[metric] > 0)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=pla_color(pla),
                        label=f"{pla} ε={eps}", markersize=5, linewidth=1.2)
                seen_any = True
        ax.set_title(f"On-disk: fetch_strategy vs {label}")
        ax.set_xlabel("fetch_strategy (0=one-by-one, 1=all-at-once, 2=sorted, 3=model-biased)")
        ax.set_ylabel(label)
        if seen_any:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "ondisk_fetch_strategy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 7: Routing comparison (pgm-index vs fiting-tree) ────────────────────
def plot_routing_comparison(rows: List[Dict], out_dir: Path):
    """IM-B: ops_s and p99 across routing modes, one canonical dataset."""
    inmem = [r for r in rows
             if r["scenario"] == "inmem"
             and r.get("routing")
             and r["threads"] == 1.0]
    if not inmem:
        return

    # Pick the dataset with the most rows
    ds_counts = defaultdict(int)
    for r in inmem:
        ds_counts[dataset_label(r)] += 1
    best_ds = max(ds_counts, key=ds_counts.get) if ds_counts else None
    inmem = [r for r in inmem if dataset_label(r) == best_ds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["ops_s", "p99_ns"],
                                   ["Throughput (ops/s)", "p99 Latency (ns)"]):
        for pla, prows in group_by(inmem, "pla").items():
            for routing, rrows in group_by(prows, "routing").items():
                pts = sorted((r["epsilon"], r[metric]) for r in rrows if r[metric] > 0)
                if pts:
                    xs, ys = zip(*pts)
                    ls = "--" if "pgm" in str(routing) else "-"
                    ax.plot(xs, ys, linestyle=ls, marker="o",
                            label=f"{pla}/{routing}", color=pla_color(pla),
                            markersize=5)
        ax.set_title(f"IM-B: Routing × PLA — {ylabel} ({best_ds})")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "routing_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 8: Segment length distribution ──────────────────────────────────────
def plot_seg_len_distribution(rows: List[Dict], out_dir: Path):
    """IM-A: seg_len_mean and seg_len_p95 vs epsilon, per (pla, dataset)."""
    pla_only = [r for r in rows
                if r["scenario"] == "pla_only"
                and r.get("seg_len_mean", 0)
                and r.get("threads") == 1.0]
    if not pla_only:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["seg_len_mean", "seg_len_p95"],
                                   ["Mean keys/segment", "p95 keys/segment"]):
        seen = _plot_grouped_lines(ax, pla_only, "epsilon", metric,
                                   group_keys=["pla", dataset_label],
                                   filter_fn=lambda r: float(r.get(metric, 0)) > 0)
        ax.set_title(f"IM-A: ε vs {ylabel}")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        if seen:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "seg_len_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 9: Parallel build overhead ──────────────────────────────────────────
def plot_parallel_build_overhead(rows: List[Dict], out_dir: Path):
    """IM-A: extra segments from parallel build, aggregated per (pla, threads)."""
    pla_only = [r for r in rows if r["scenario"] == "pla_only"]
    if not pla_only:
        return

    baseline: Dict = {}
    for r in pla_only:
        if r["threads"] == 1.0 and r["seg_cnt"] > 0:
            key = (r["pla"], r["epsilon"], dataset_label(r))
            baseline[key] = r["seg_cnt"]

    # Aggregate overhead per (pla, threads) across all epsilons and datasets
    overheads: Dict[str, List[float]] = defaultdict(list)
    for r in pla_only:
        if r["threads"] <= 1.0 or r["seg_cnt"] <= 0:
            continue
        bk = (r["pla"], r["epsilon"], dataset_label(r))
        base = baseline.get(bk, 0)
        if base > 0:
            overhead = (r["seg_cnt"] - base) / base * 100.0
            key = (r["pla"], int(r["threads"]))
            overheads[key].append(overhead)

    fig, ax = plt.subplots(figsize=(8, 5))
    seen_any = False
    for pla in ["optimal", "swing", "greedy"]:
        pts = []
        for (p, threads), vals in sorted(overheads.items()):
            if p == pla and vals:
                pts.append((threads, sum(vals) / len(vals)))
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, marker="^", label=pla, color=pla_color(pla))
            seen_any = True

    if not seen_any:
        ax.text(0.5, 0.5, "No multi-thread pla_only data",
                ha="center", transform=ax.transAxes)
    else:
        ax.legend()
    ax.set_title("IM-A: Parallel Build Overhead (mean extra segments %)")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Extra segments vs 1-thread (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "parallel_build_overhead.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 10: Retrain impact (DW-A/B) ─────────────────────────────────────────
def plot_retrain_impact(rows: List[Dict], out_dir: Path):
    """DW-A/B: retrain_ms and p99 latency, grouped by workload."""
    dynamic = [r for r in rows
               if r["scenario"] == "dynamic" and r["build_ms"] > 0]
    if not dynamic:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: retrain total time vs epsilon, one line per (pla, workload, dataset)
    ax = axes[0]
    for wl in sorted(set(r.get("workload", "") for r in dynamic)):
        wr = [r for r in dynamic if r.get("workload") == wl]
        for pla, prows in group_by(wr, "pla").items():
            # Average retrain_ms across datasets at each epsilon
            eps_groups = defaultdict(list)
            for r in prows:
                rt = r.get("retrain_ms", 0)
                if rt > 0:
                    eps_groups[int(r["epsilon"])].append(rt)
            pts = sorted((eps, sum(vals)/len(vals)) for eps, vals in eps_groups.items())
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=pla_color(pla),
                        label=f"{pla}/{wl}", markersize=5, linewidth=1.2)
    ax.set_title("DW-A: ε vs Retrain Time (by workload)")
    ax.set_xlabel("ε (epsilon)")
    ax.set_ylabel("retrain_ms (cumulative)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: p99 lookup latency vs epsilon, by workload
    ax = axes[1]
    for wl in sorted(set(r.get("workload", "") for r in dynamic)):
        wr = [r for r in dynamic if r.get("workload") == wl]
        for pla, prows in group_by(wr, "pla").items():
            eps_groups = defaultdict(list)
            for r in prows:
                if r["p99_ns"] > 0:
                    eps_groups[int(r["epsilon"])].append(r["p99_ns"])
            pts = sorted((eps, sum(vals)/len(vals)) for eps, vals in eps_groups.items())
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="D", color=pla_color(pla),
                        label=f"{pla}/{wl}", markersize=5, linewidth=1.2)
    ax.set_title("DW-B: ε vs p99 Lookup Latency (by workload)")
    ax.set_xlabel("ε (epsilon)")
    ax.set_ylabel("p99 latency (ns)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "retrain_impact.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 11: On-disk Rp (pages/query) vs epsilon ─────────────────────────────
def plot_ondisk_rp(rows: List[Dict], out_dir: Path):
    """OD-A/B: io_pages_mean vs epsilon, canonical slice."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk"
              and r.get("io_pages_mean", 0) > 0
              and r.get("granularity", "item") == "item"
              and str(r.get("page_align", "")).lower() in ("false", "0", "")
              and r.get("fetch_strategy", 0) == 0.0
              and r.get("workload", "readonly") == "readonly"]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["io_pages_mean", "ops_s"],
                                   ["Mean pages/query (Rp)", "Throughput (ops/s)"]):
        seen = _plot_grouped_lines(ax, ondisk, "epsilon", metric,
                                   group_keys=["pla", dataset_label],
                                   filter_fn=lambda r: float(r.get(metric, 0)) > 0)
        ax.set_title(f"OD-A/B: ε vs {ylabel} (item, no-align, readonly)")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        if seen:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "ondisk_rp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 12: G3 page-align benefit (OD-E) ────────────────────────────────────
def plot_page_align_benefit(rows: List[Dict], out_dir: Path):
    """OD-E: io_pages_mean and ops_s with page-align ON vs OFF, canonical slice."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk"
              and r.get("page_align") is not None
              and r.get("granularity", "item") == "item"
              and r.get("fetch_strategy", 0) == 0.0
              and r.get("workload", "readonly") == "readonly"]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["io_pages_mean", "ops_s"],
                                   ["Mean pages/query", "Throughput (ops/s)"]):
        for pla, prows in group_by(ondisk, "pla").items():
            for align in [False, True]:
                arows = [r for r in prows
                         if str(r.get("page_align", "false")).lower() in
                            (("true", "1") if align else ("false", "0", ""))]
                pts = sorted((r["epsilon"], float(r.get(metric, 0)))
                             for r in arows if float(r.get(metric, 0)) > 0)
                if pts:
                    xs, ys = zip(*pts)
                    ls = "-" if not align else "--"
                    lbl = f"{pla}/{'aligned' if align else 'raw'}"
                    ax.plot(xs, ys, linestyle=ls, marker="s",
                            label=lbl, color=pla_color(pla), markersize=5)
        ax.set_title(f"OD-E: G3 Page-Align Effect ({ylabel})")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "page_align_benefit.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 13: RSS memory across scenarios ────────────────────────────────────
def plot_rss(rows: List[Dict], out_dir: Path):
    """ε vs rss_mb, facet only by scenarios that have RSS data."""
    scenarios = ["pla_only", "inmem", "dynamic", "ondisk"]
    active = [(s, [r for r in rows if r["scenario"] == s and r.get("rss_mb", 0) > 0])
              for s in scenarios]
    active = [(s, sr) for s, sr in active if sr]
    if not active:
        return
    n = len(active)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows),
                             squeeze=False)
    for idx, (scenario, sr) in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        seen = _plot_grouped_lines(ax, sr, "epsilon", "rss_mb",
                                   group_keys=["pla", dataset_label])
        ax.set_title(f"{scenario}: ε vs RSS")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel("RSS (MB)")
        if seen:
            ax.legend(fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
    _hide_unused_subplots(axes, n)
    plt.tight_layout()
    out = out_dir / "rss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 14: Index size across scenarios ────────────────────────────────────
def plot_bytes_index(rows: List[Dict], out_dir: Path):
    """ε vs bytes_index, facet only by scenarios that have data."""
    scenarios = ["pla_only", "inmem", "dynamic", "ondisk"]
    active = [(s, [r for r in rows if r["scenario"] == s and r.get("bytes_index", 0) > 0])
              for s in scenarios]
    active = [(s, sr) for s, sr in active if sr]
    if not active:
        return
    n = len(active)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows),
                             squeeze=False)
    for idx, (scenario, sr) in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        seen = _plot_grouped_lines(ax, sr, "epsilon", "bytes_index",
                                   group_keys=["pla", dataset_label])
        ax.set_title(f"{scenario}: ε vs Index Size")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel("Index size (bytes)")
        ax.set_yscale("log")
        if seen:
            ax.legend(fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
    _hide_unused_subplots(axes, n)
    plt.tight_layout()
    out = out_dir / "bytes_index.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 15: G4 compress effect (ondisk) ────────────────────────────────────
def _bool_val(r: Dict, key: str) -> bool:
    v = str(r.get(key, "")).lower()
    return v in ("true", "1")


def plot_ondisk_compress(rows: List[Dict], out_dir: Path):
    """OD-B: compress on/off comparison — ops_s and bytes_index."""
    ondisk = [r for r in rows if r["scenario"] == "ondisk"]
    has_compress = any(_bool_val(r, "compress") for r in ondisk)
    if not has_compress:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["ops_s", "bytes_index"],
                                   ["Throughput (ops/s)", "Index size (bytes)"]):
        seen_any = False
        for pla in PLA_COLORS:
            for comp_flag, comp_label, ls in [("false", "uncompressed", "-"),
                                               ("true", "compressed", "--")]:
                prows = [r for r in ondisk
                         if r["pla"] == pla and _bool_val(r, "compress") == (comp_flag == "true")]
                # Use bytes_compressed for compressed size when available
                if comp_flag == "true" and metric == "bytes_index":
                    pts = sorted((r["epsilon"], r.get("bytes_compressed", 0) or r["bytes_index"])
                                 for r in prows if (r.get("bytes_compressed", 0) or r["bytes_index"]) > 0)
                else:
                    pts = sorted((r["epsilon"], r[metric]) for r in prows if r[metric] > 0)
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, linestyle=ls, marker="o", color=pla_color(pla),
                            label=f"{pla} {comp_label}", markersize=5)
                    seen_any = True
        ax.set_title(f"OD-B: ε vs {ylabel} (compress on/off)")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        if seen_any:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "ondisk_compress.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 16: Dynamic throughput by workload (DW-B) ──────────────────────────
def plot_dynamic_throughput(rows: List[Dict], out_dir: Path):
    """DW-B: ε vs ops_s for dynamic, grouped by (pla, workload)."""
    dynamic = [r for r in rows
               if r["scenario"] == "dynamic" and r["ops_s"] > 0]
    if not dynamic:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    seen_any = False
    for wl in sorted(set(r.get("workload", "") for r in dynamic)):
        wr = [r for r in dynamic if r.get("workload") == wl]
        for pla in PLA_COLORS:
            prows = [r for r in wr if r["pla"] == pla]
            eps_groups: Dict[int, List[float]] = defaultdict(list)
            for r in prows:
                eps_groups[int(r["epsilon"])].append(r["ops_s"])
            pts = sorted((eps, sum(vals) / len(vals)) for eps, vals in eps_groups.items())
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=pla_color(pla),
                        label=f"{pla} / {wl}", markersize=5, linewidth=1.2)
                seen_any = True
    ax.set_title("DW-B: ε vs Throughput by Workload")
    ax.set_xlabel("ε (epsilon)")
    ax.set_ylabel("Throughput (ops/s)")
    if seen_any:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "dynamic_throughput.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 17: Slope / intercept std (IM-A) ───────────────────────────────────
def plot_slope_intercept(rows: List[Dict], out_dir: Path):
    """IM-A: ε vs slope_std and intercept_std for pla_only, single-thread."""
    pla_only = [r for r in rows
                if r["scenario"] == "pla_only"
                and r.get("threads", 1) == 1.0]
    if not pla_only:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["slope_std", "intercept_std"],
                                   ["Slope std", "Intercept std"]):
        seen = _plot_grouped_lines(ax, pla_only, "epsilon", metric,
                                   group_keys=["pla", dataset_label],
                                   filter_fn=lambda r: float(r.get(metric, 0)) > 0)
        ax.set_title(f"IM-A: ε vs {ylabel}")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        if seen:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "slope_intercept.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 18: Microarch — IPC + branch miss rate (IM-B) ──────────────────────
def plot_microarch(rows: List[Dict], out_dir: Path):
    """IM-B: ε vs IPC and branch_miss_rate for inmem, canonical slice."""
    inmem = [r for r in rows
             if r["scenario"] == "inmem"
             and r.get("threads", 1) == 1.0
             and r.get("ipc", 0) > 0]
    if not inmem:
        return

    # Compute branch miss rate on the fly
    for r in inmem:
        branches = r.get("branches", 0)
        r["branch_miss_rate"] = (r.get("branch_misses", 0) / branches * 100.0) if branches else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["ipc", "branch_miss_rate"],
                                   ["IPC (instructions/cycle)", "Branch miss rate (%)"]):
        seen_any = False
        for pla in PLA_COLORS:
            eps_groups: Dict[int, List[float]] = defaultdict(list)
            for r in inmem:
                if r["pla"] == pla and float(r.get(metric, 0)) > 0:
                    eps_groups[int(r["epsilon"])].append(float(r[metric]))
            pts = sorted((eps, sum(vals) / len(vals)) for eps, vals in eps_groups.items())
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=pla_color(pla),
                        label=pla, markersize=5, linewidth=1.5)
                seen_any = True
        ax.set_title(f"IM-B: ε vs {ylabel}")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        if seen_any:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "microarch.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 19: Granularity item vs page (OD-D) ────────────────────────────────
def plot_ondisk_granularity(rows: List[Dict], out_dir: Path):
    """OD-D: item vs page granularity comparison for ondisk."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk"
              and r.get("granularity")
              and str(r.get("page_align", "")).lower() in ("false", "0", "")
              and r.get("fetch_strategy", 0) == 1.0
              and r.get("workload", "readonly") == "readonly"]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["io_pages_mean", "ops_s"],
                                   ["Mean pages/query", "Throughput (ops/s)"]):
        seen_any = False
        for gran, gran_label, ls in [("item", "item", "-"), ("page", "page", "--")]:
            gr = [r for r in ondisk if r.get("granularity") == gran]
            for pla in PLA_COLORS:
                prows = [r for r in gr if r["pla"] == pla]
                pts = sorted((r["epsilon"], r[metric]) for r in prows if r[metric] > 0)
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, linestyle=ls, marker="s", color=pla_color(pla),
                            label=f"{pla} {gran_label}", markersize=5)
                    seen_any = True
        ax.set_title(f"OD-D: ε vs {ylabel} (item vs page)")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        if seen_any:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "ondisk_granularity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 20: Fetch strategy × threads (OD-C) ────────────────────────────────
def plot_ondisk_fetch_threads(rows: List[Dict], out_dir: Path):
    """OD-C: fetch_strategy vs ops_s, faceted by thread count."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk"
              and r.get("granularity", "item") == "item"
              and str(r.get("page_align", "")).lower() in ("false", "0", "")
              and r.get("workload", "readonly") == "readonly"
              and r.get("fetch_strategy", -1) >= 0
              and r["ops_s"] > 0]
    if not ondisk:
        return

    thread_vals = sorted(set(int(r.get("threads", 1)) for r in ondisk))
    n_threads = len(thread_vals)
    if n_threads <= 1:
        return  # single-thread only — plot 6 already covers this

    ncols = min(3, n_threads)
    nrows = (n_threads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)
    for idx, threads in enumerate(thread_vals):
        ax = axes[idx // ncols][idx % ncols]
        tr = [r for r in ondisk if int(r.get("threads", 1)) == threads]
        seen_any = False
        for pla in PLA_COLORS:
            prows = [r for r in tr if r["pla"] == pla]
            pts = sorted((r["fetch_strategy"], r["ops_s"]) for r in prows if r["ops_s"] > 0)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=pla_color(pla),
                        label=pla, markersize=5)
                seen_any = True
        ax.set_title(f"threads={threads}: fetch_strategy vs ops/s")
        ax.set_xlabel("fetch_strategy")
        ax.set_ylabel("ops/s")
        if seen_any:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    # Hide unused subplots
    for idx in range(n_threads, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    plt.tight_layout()
    out = out_dir / "ondisk_fetch_threads.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 21: Target Rp iso-Rp analysis (OD-A) ───────────────────────────────
def plot_target_rp(rows: List[Dict], out_dir: Path):
    """OD-A iso-Rp: target_rp vs epsilon and seg_cnt per PLA."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk"
              and r.get("target_rp", 0) > 0]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["epsilon", "seg_cnt"],
                                   ["ε needed", "Segment count"]):
        seen = _plot_grouped_lines(ax, ondisk, "target_rp", metric,
                                   group_keys=["pla", dataset_label])
        ax.set_title(f"OD-A iso-Rp: target Rp vs {ylabel}")
        ax.set_xlabel("target_rp (target pages/query)")
        ax.set_ylabel(ylabel)
        if seen:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "target_rp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 22: Hybrid workloads (OD-F) ─────────────────────────────────────────
def plot_ondisk_workloads(rows: List[Dict], out_dir: Path):
    """OD-F: ondisk workload comparison — ops_s and io_pages_mean per PLA."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk"
              and r.get("workload") in ("readonly", "insert", "hybrid")
              and r.get("granularity", "item") == "item"
              and str(r.get("page_align", "")).lower() in ("false", "0", "")]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel in zip(axes,
                                   ["ops_s", "io_pages_mean"],
                                   ["Throughput (ops/s)", "Mean pages/query"]):
        seen_any = False
        for wl in sorted(set(r.get("workload", "") for r in ondisk)):
            wr = [r for r in ondisk if r.get("workload") == wl]
            for pla in PLA_COLORS:
                prows = [r for r in wr if r["pla"] == pla]
                pts = sorted((r["epsilon"], r[metric]) for r in prows if r[metric] > 0)
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, marker="o", color=pla_color(pla),
                            label=f"{pla} / {wl}", markersize=5, linewidth=1.2)
                    seen_any = True
        ax.set_title(f"OD-F: ε vs {ylabel} by Workload")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(ylabel)
        if seen_any:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "ondisk_workloads.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 23: Retrain count (DW-A) ────────────────────────────────────────────
def plot_retrain_count(rows: List[Dict], out_dir: Path):
    """DW-A: ε vs retrain_count for dynamic, grouped by (pla, workload)."""
    dynamic = [r for r in rows
               if r["scenario"] == "dynamic"
               and r.get("retrain_count", 0) > 0]
    if not dynamic:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    seen_any = False
    for wl in sorted(set(r.get("workload", "") for r in dynamic)):
        wr = [r for r in dynamic if r.get("workload") == wl]
        for pla in PLA_COLORS:
            prows = [r for r in wr if r["pla"] == pla]
            eps_groups: Dict[int, List[float]] = defaultdict(list)
            for r in prows:
                eps_groups[int(r["epsilon"])].append(r["retrain_count"])
            pts = sorted((eps, sum(vals) / len(vals)) for eps, vals in eps_groups.items())
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="s", color=pla_color(pla),
                        label=f"{pla} / {wl}", markersize=5, linewidth=1.2)
                seen_any = True
    ax.set_title("DW-A: ε vs Retrain Count by Workload")
    ax.set_xlabel("ε (epsilon)")
    ax.set_ylabel("Retrain count")
    if seen_any:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "retrain_count.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 24: PLA build scaling (threads vs build_ms) ────────────────────────
def plot_pla_build_scaling(rows: List[Dict], out_dir: Path):
    """IM-A: threads vs build_ms for pla_only, canonical epsilon."""
    pla_only = [r for r in rows
                if r["scenario"] == "pla_only"
                and r["build_ms"] > 0]
    if not pla_only:
        return

    # Pick the epsilon value with the most data
    eps_counts: Dict[int, int] = defaultdict(int)
    for r in pla_only:
        eps_counts[int(r["epsilon"])] += 1
    best_eps = max(eps_counts, key=eps_counts.get) if eps_counts else None
    if best_eps is None:
        return

    sr = [r for r in pla_only if int(r["epsilon"]) == best_eps]

    fig, ax = plt.subplots(figsize=(10, 6))
    seen = _plot_grouped_lines(ax, sr, "threads", "build_ms",
                               group_keys=["pla", dataset_label])
    ax.set_title(f"IM-A: threads vs Build Time (ε={best_eps})")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Build time (ms)")
    if seen:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "pla_build_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# In-Memory Space-Time Tradeoff & Perf-Counter Analysis
# ══════════════════════════════════════════════════════════════════════════════

def _perf_rate(rows: List[Dict]) -> List[Dict]:
    """Add derived perf-counter rates in-place."""
    for r in rows:
        instr = r.get("instructions", 0) or 1
        queries = r.get("n_keys", 0) or 1
        r["cache_miss_rate"] = (r.get("cache_misses", 0) or 0) / instr
        r["ipc"] = instr / ((r.get("cycles", 0) or 0) or 1)
        r["branch_miss_rate"] = ((r.get("branch_misses", 0) or 0) / instr) * 100.0
        r["instr_per_lookup"] = instr / queries
    return rows


# ── Plot 25: Space-Time Tradeoff (inmem) ────────────────────────────────────
def plot_inmem_spacetime(rows: List[Dict], out_dir: Path):
    """seg_cnt vs ops_s and seg_cnt vs p99_ns — one subplot per routing.

    Each line connects (seg_cnt, metric) points of one PLA across the epsilon
    sweep.  This is the core Pareto-frontier view: fewer segments (left) +
    higher throughput (up) = strictly better.
    """
    inmem = [r for r in rows if r["scenario"] == "inmem" and r["ops_s"] > 0]
    if not inmem:
        return

    routings = sorted(set(r.get("routing", "") for r in inmem if r.get("routing")))
    if not routings:
        return

    fig, axes = plt.subplots(2, len(routings), figsize=(7 * len(routings), 11),
                             squeeze=False)
    for col, routing in enumerate(routings):
        rr = [r for r in inmem if r.get("routing") == routing]

        # Top row: seg_cnt vs ops_s
        ax = axes[0][col]
        seen = _plot_grouped_lines(ax, rr, "seg_cnt", "ops_s",
                                   group_keys=["pla"],
                                   marker_fn=lambda i: "o")
        ax.set_title(f"{routing}: seg_cnt vs Throughput")
        ax.set_xlabel("Segment count")
        ax.set_ylabel("Throughput (ops/s)")
        ax.set_xscale("log")
        if seen:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: seg_cnt vs p99_ns
        ax = axes[1][col]
        seen = _plot_grouped_lines(ax, rr, "seg_cnt", "p99_ns",
                                   group_keys=["pla"],
                                   marker_fn=lambda i: "o")
        ax.set_title(f"{routing}: seg_cnt vs p99 Latency")
        ax.set_xlabel("Segment count")
        ax.set_ylabel("p99 latency (ns)")
        ax.set_xscale("log")
        if seen:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "inmem_spacetime.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 26: ε vs Perf Counters (inmem) ─────────────────────────────────────
def plot_inmem_perfcounters(rows: List[Dict], out_dir: Path):
    """ε vs cache_miss_rate, IPC, branch_miss_rate — one row per routing.

    Shows how PLA choice affects microarchitectural behaviour as epsilon
    (and thus segment count) changes.  Explains the WHY behind the space-time
    tradeoff curves.
    """
    inmem = _perf_rate([r for r in rows if r["scenario"] == "inmem"
                        and r.get("instructions", 0) > 0])
    if not inmem:
        return

    routings = sorted(set(r.get("routing", "") for r in inmem if r.get("routing")))
    if not routings:
        return

    metrics = [
        ("cache_miss_rate", "Cache-miss rate\n(misses / instruction)"),
        ("ipc",             "IPC\n(instructions / cycle)"),
        ("branch_miss_rate","Branch-miss rate (%)\n(branch misses / instr × 100)"),
    ]

    fig, axes = plt.subplots(len(metrics), len(routings),
                             figsize=(7 * len(routings), 5 * len(metrics)),
                             squeeze=False)
    for col, routing in enumerate(routings):
        rr = [r for r in inmem if r.get("routing") == routing]
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            seen = _plot_grouped_lines(ax, rr, "epsilon", metric,
                                       group_keys=["pla"],
                                       marker_fn=lambda i: "o")
            ax.set_title(f"{routing}: ε vs {ylabel.split(chr(10))[0]}")
            ax.set_xlabel("ε (epsilon)")
            ax.set_ylabel(ylabel)
            if seen:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "inmem_perfcounters.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 27: Perf Counters vs Performance (scatter) ─────────────────────────
def plot_inmem_perf_scatter(rows: List[Dict], out_dir: Path):
    """Scatter: cache_miss_rate vs ops_s, branch_miss_rate vs p99_ns.

    Each point is one (pla, epsilon) combination.  Marker = routing.
    Reveals correlation between microarchitectural counters and end-to-end
    performance.
    """
    inmem = _perf_rate([r for r in rows if r["scenario"] == "inmem"
                        and r.get("instructions", 0) > 0
                        and r["ops_s"] > 0])
    if not inmem:
        return

    routings = sorted(set(r.get("routing", "") for r in inmem if r.get("routing")))
    routing_markers = {r: ["o", "s", "^", "D"][i]
                       for i, r in enumerate(routings)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: cache_miss_rate vs ops_s
    ax = axes[0]
    for routing in routings:
        rr = [r for r in inmem if r.get("routing") == routing]
        for pla in PLA_COLORS:
            pr = [r for r in rr if r["pla"] == pla]
            pts = [(r["cache_miss_rate"], r["ops_s"]) for r in pr]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.scatter(xs, ys, marker=routing_markers[routing],
                          color=pla_color(pla), alpha=0.7, s=40,
                          label=f"{pla} / {routing}")
                # Connect with faint line to show ε sweep direction
                ax.plot(xs, ys, color=pla_color(pla), alpha=0.25, linewidth=0.8)
    ax.set_title("Cache-miss rate vs Throughput")
    ax.set_xlabel("Cache-miss rate (misses / instruction)")
    ax.set_ylabel("Throughput (ops/s)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: branch_miss_rate vs p99_ns
    ax = axes[1]
    for routing in routings:
        rr = [r for r in inmem if r.get("routing") == routing]
        for pla in PLA_COLORS:
            pr = [r for r in rr if r["pla"] == pla]
            pts = [(r["branch_miss_rate"], r["p99_ns"]) for r in pr]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.scatter(xs, ys, marker=routing_markers[routing],
                          color=pla_color(pla), alpha=0.7, s=40,
                          label=f"{pla} / {routing}")
                ax.plot(xs, ys, color=pla_color(pla), alpha=0.25, linewidth=0.8)
    ax.set_title("Branch-miss rate vs p99 Latency")
    ax.set_xlabel("Branch-miss rate (%)")
    ax.set_ylabel("p99 latency (ns)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "inmem_perf_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 28: Instructions-per-lookup diagnostic ─────────────────────────────
def plot_inmem_instr_per_lookup(rows: List[Dict], out_dir: Path):
    """seg_cnt vs instructions-per-lookup, one subplot per routing.

    Tests: does more segments → more instructions per lookup?  If the
    routing layer absorbs segment count differences, this relationship
    should be weak.
    """
    inmem = _perf_rate([r for r in rows if r["scenario"] == "inmem"
                        and r.get("instructions", 0) > 0])
    if not inmem:
        return

    routings = sorted(set(r.get("routing", "") for r in inmem if r.get("routing")))
    if not routings:
        return

    fig, axes = plt.subplots(1, len(routings), figsize=(7 * len(routings), 5),
                             squeeze=False)
    for col, routing in enumerate(routings):
        ax = axes[0][col]
        rr = [r for r in inmem if r.get("routing") == routing]
        seen = _plot_grouped_lines(ax, rr, "seg_cnt", "instr_per_lookup",
                                   group_keys=["pla"],
                                   marker_fn=lambda i: "o")
        ax.set_title(f"{routing}: seg_cnt vs Instructions/Lookup")
        ax.set_xlabel("Segment count")
        ax.set_ylabel("Instructions per lookup")
        ax.set_xscale("log")
        if seen:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "inmem_instr_per_lookup.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate benchmark plots")
    ap.add_argument("--input",  required=True, help="Path to results.csv")
    ap.add_argument("--output", required=True, help="Output directory for PNGs")
    args = ap.parse_args()

    if not HAS_MPL:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return

    csv_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    # Detect single-dataset mode — when all rows share the same dataset we
    # drop the dataset name from legend labels for much cleaner output.
    global _SINGLE_DATASET
    datasets = {_norm_dataset(r) for r in rows}
    _SINGLE_DATASET = len(datasets) <= 1
    if _SINGLE_DATASET:
        ds_name = next(iter(datasets)) if datasets else "unknown"
        print(f"Single-dataset mode: \"{ds_name}\" — dataset names omitted from legends")

    plot_epsilon_seg_cnt(rows, out_dir)
    plot_epsilon_build_ms(rows, out_dir)
    plot_threads_throughput(rows, out_dir)
    plot_latency_cdf(rows, out_dir)
    plot_cache_miss_rate(rows, out_dir)
    plot_ondisk_fetch(rows, out_dir)
    plot_routing_comparison(rows, out_dir)
    plot_seg_len_distribution(rows, out_dir)
    plot_parallel_build_overhead(rows, out_dir)
    plot_retrain_impact(rows, out_dir)
    plot_ondisk_rp(rows, out_dir)
    plot_page_align_benefit(rows, out_dir)
    plot_rss(rows, out_dir)
    plot_bytes_index(rows, out_dir)
    plot_ondisk_compress(rows, out_dir)
    plot_dynamic_throughput(rows, out_dir)
    plot_slope_intercept(rows, out_dir)
    plot_microarch(rows, out_dir)
    plot_ondisk_granularity(rows, out_dir)
    plot_ondisk_fetch_threads(rows, out_dir)
    plot_target_rp(rows, out_dir)
    plot_ondisk_workloads(rows, out_dir)
    plot_retrain_count(rows, out_dir)
    plot_pla_build_scaling(rows, out_dir)
    plot_inmem_spacetime(rows, out_dir)
    plot_inmem_perfcounters(rows, out_dir)
    plot_inmem_perf_scatter(rows, out_dir)
    plot_inmem_instr_per_lookup(rows, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
