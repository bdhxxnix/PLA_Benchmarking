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


def dataset_label(r: Dict) -> str:
    """Short readable dataset label: 'uniform_100K', 'sosd_fb_200M', etc."""
    ds = str(r.get("dataset", ""))
    ds = ds.split("/")[-1] if "/" in ds else ds
    if not ds:
        ds = "unknown"
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

    Lines are labelled "{pla} / {dataset}" so the legend stays readable.
    """
    if filter_fn:
        rows = [r for r in rows if filter_fn(r)]

    # Build composite groups
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        parts = [str(r.get(k, "")) for k in group_keys]
        label = " / ".join(p for p in parts if p)
        groups[label].append(r)

    seen_any = False
    for idx, (label, prows) in enumerate(sorted(groups.items())):
        pts = sorted((r[x_key], r[y_key]) for r in prows if r[y_key] > 0)
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

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
