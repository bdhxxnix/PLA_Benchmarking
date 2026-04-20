#!/usr/bin/env python3
"""
tools/viz/plots.py
Generate benchmark plots from aggregated CSV.

Charts produced:
  1. epsilon vs seg_cnt       (line per pla, facet by scenario)
  2. epsilon vs build_ms      (line per pla)
  3. threads vs throughput    (line per pla, facet by scenario)
  4. latency CDF              (one per scenario/pla/epsilon combo)
  5. cache-miss rate          (bar: pla x epsilon)
  6. fetch_strategy vs p99/ops_s  (ondisk only)

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
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available; skipping plots.", file=sys.stderr)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # Cast numeric fields.
            for k in ("epsilon","threads","seg_cnt","bytes_index","fetch_strategy",
                      "build_ms","ops_s","p50_ns","p95_ns","p99_ns",
                      "cache_misses","instructions","cycles","cache_miss_rate","ipc"):
                try: row[k] = float(row[k])
                except (ValueError, KeyError): row[k] = 0.0
            rows.append(row)
    return rows


# ── Colour palette ────────────────────────────────────────────────────────────
PLA_COLORS = {"optimal": "#1f77b4", "swing": "#ff7f0e", "greedy": "#2ca02c"}

def pla_color(pla: str) -> str:
    return PLA_COLORS.get(pla, "#999999")


# ── Helper: group rows ────────────────────────────────────────────────────────
def group_by(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    g: Dict[str, List] = defaultdict(list)
    for r in rows:
        g[r.get(key, "")].append(r)
    return dict(g)


# ── Plot 1: ε vs seg_cnt ─────────────────────────────────────────────────────
def plot_epsilon_seg_cnt(rows: List[Dict], out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, scenario in zip(axes, ["pla_only", "inmem"]):
        sr = [r for r in rows if r["scenario"] == scenario]
        if not sr:
            ax.set_title(scenario + " (no data)")
            continue
        for pla, prows in group_by(sr, "pla").items():
            pts = sorted((r["epsilon"], r["seg_cnt"]) for r in prows if r["seg_cnt"] > 0)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=pla, color=pla_color(pla))
        ax.set_title(f"{scenario}: ε vs seg_cnt")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel("Number of segments")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "epsilon_vs_seg_cnt.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 2: ε vs build_ms ────────────────────────────────────────────────────
def plot_epsilon_build_ms(rows: List[Dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for pla, prows in group_by(rows, "pla").items():
        pts = sorted((r["epsilon"], r["build_ms"]) for r in prows if r["build_ms"] > 0)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="s", label=pla, color=pla_color(pla))
    ax.set_title("ε vs Build Time")
    ax.set_xlabel("ε (epsilon)"); ax.set_ylabel("build_ms")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "epsilon_vs_build_ms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 3: threads vs throughput ────────────────────────────────────────────
def plot_threads_throughput(rows: List[Dict], out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, scenario in zip(axes, ["inmem", "dynamic"]):
        sr = [r for r in rows if r["scenario"] == scenario and r["ops_s"] > 0]
        if not sr:
            ax.set_title(scenario + " (no data)")
            continue
        for pla, prows in group_by(sr, "pla").items():
            pts = sorted((r["threads"], r["ops_s"]) for r in prows)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="^", label=pla, color=pla_color(pla))
        ax.set_title(f"{scenario}: threads vs ops/s")
        ax.set_xlabel("Threads"); ax.set_ylabel("ops/s")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "threads_vs_throughput.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 4: Latency CDF ──────────────────────────────────────────────────────
def plot_latency_cdf(rows: List[Dict], out_dir: Path):
    # We only have p50/p95/p99 aggregates; synthesise a 3-point CDF.
    fig, ax = plt.subplots(figsize=(8, 5))
    seen_any = False
    for pla, prows in group_by(rows, "pla").items():
        for r in prows:
            if r["p50_ns"] == 0:
                continue
            pts = [(r["p50_ns"], 50), (r["p95_ns"], 95), (r["p99_ns"], 99)]
            xs, ys = zip(*sorted(pts))
            lbl = f"{r['scenario']}/{pla}/ε={int(r['epsilon'])}"
            ax.plot(xs, ys, marker="o", label=lbl, color=pla_color(pla),
                    alpha=0.7, linewidth=1)
            seen_any = True
    if not seen_any:
        ax.text(0.5, 0.5, "No latency data", ha="center", transform=ax.transAxes)
    ax.set_title("Latency CDF (p50/p95/p99)")
    ax.set_xlabel("Latency (ns)"); ax.set_ylabel("Percentile")
    ax.legend(fontsize=6, loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "latency_cdf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 5: Cache-miss rate ──────────────────────────────────────────────────
def plot_cache_miss_rate(rows: List[Dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epsilons = sorted(set(int(r["epsilon"]) for r in rows if r["cache_miss_rate"] > 0))
    plas     = list(PLA_COLORS.keys())
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
                vals.append(sum(matched)/len(matched) if matched else 0)
            ax.bar([xi + i*width for xi in x], vals, width, label=pla,
                   color=pla_color(pla), alpha=0.8)
        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.set_xlabel("ε"); ax.set_ylabel("Cache-miss rate (misses/instruction)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Cache-Miss Rate by ε and PLA")
    plt.tight_layout()
    out = out_dir / "cache_miss_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 6: fetch_strategy vs p99 / ops_s (ondisk) ──────────────────────────
def plot_ondisk_fetch(rows: List[Dict], out_dir: Path):
    ondisk = [r for r in rows if r["scenario"] == "ondisk"]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes,
                                  ["p99_ns", "ops_s"],
                                  ["p99 latency (ns)", "Throughput (ops/s)"]):
        for pla, prows in group_by(ondisk, "pla").items():
            pts = sorted((r["fetch_strategy"], r[metric]) for r in prows if r[metric] > 0)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="D", label=pla, color=pla_color(pla))
        ax.set_title(f"On-disk: fetch_strategy vs {label}")
        ax.set_xlabel("fetch_strategy (0=one-by-one, 1=all-at-once, 2=sorted, 3=model-biased)")
        ax.set_ylabel(label)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "ondisk_fetch_strategy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 7: Routing comparison (pgm-index vs fiting-tree) ────────────────────
def plot_routing_comparison(rows: List[Dict], out_dir: Path):
    """IM-B: compare ops_s and p99 across routing modes per PLA algorithm."""
    inmem = [r for r in rows if r["scenario"] == "inmem" and r.get("routing")]
    if not inmem:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes,
                                  ["ops_s", "p99_ns"],
                                  ["Throughput (ops/s)", "p99 Latency (ns)"]):
        for pla, prows in group_by(inmem, "pla").items():
            for routing, rrows in group_by(prows, "routing").items():
                pts = sorted((r["epsilon"], r[metric]) for r in rrows
                             if r[metric] > 0 and r["threads"] == 1.0)
                if pts:
                    xs, ys = zip(*pts)
                    ls = "--" if "pgm" in str(routing) else "-"
                    ax.plot(xs, ys, linestyle=ls, marker="o",
                            label=f"{pla}/{routing}", color=pla_color(pla))
        ax.set_title(f"IM-B: Routing × PLA  ({label})")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "routing_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 8: Segment length distribution (box approximation) ──────────────────
def plot_seg_len_distribution(rows: List[Dict], out_dir: Path):
    """IM-A: seg_len_mean and seg_len_p95 vs epsilon per PLA."""
    pla_only = [r for r in rows
                if r["scenario"] == "pla_only" and r.get("seg_len_mean", 0)]
    if not pla_only:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes,
                                  ["seg_len_mean", "seg_len_p95"],
                                  ["Mean key span", "p95 key span"]):
        for pla, prows in group_by(pla_only, "pla").items():
            pts = sorted((r["epsilon"], float(r.get(metric, 0)))
                         for r in prows
                         if float(r.get(metric, 0)) > 0
                         and r.get("threads") == 1.0)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=pla, color=pla_color(pla))
        ax.set_title(f"IM-A: ε vs {label}")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "seg_len_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 9: Parallel build overhead (threads vs seg_cnt relative to 1T) ──────
def plot_parallel_build_overhead(rows: List[Dict], out_dir: Path):
    """IM-A: shows extra segments produced by parallel build as thread count rises."""
    pla_only = [r for r in rows if r["scenario"] == "pla_only"]
    if not pla_only:
        return

    # Find baseline (threads=1) seg_cnt per (pla, epsilon, dataset).
    baseline: Dict = {}
    for r in pla_only:
        if r["threads"] == 1.0 and r["seg_cnt"] > 0:
            key = (r["pla"], r["epsilon"], r.get("dataset", ""))
            baseline[key] = r["seg_cnt"]

    fig, ax = plt.subplots(figsize=(8, 5))
    seen_any = False
    for pla, prows in group_by(pla_only, "pla").items():
        pts = []
        for r in prows:
            if r["threads"] <= 1.0 or r["seg_cnt"] <= 0:
                continue
            bk = (r["pla"], r["epsilon"], r.get("dataset", ""))
            base = baseline.get(bk, 0)
            if base > 0:
                overhead = (r["seg_cnt"] - base) / base * 100.0
                pts.append((r["threads"], overhead))
        if pts:
            pts_sorted = sorted(pts)
            xs, ys = zip(*pts_sorted)
            ax.plot(xs, ys, marker="^", label=pla, color=pla_color(pla))
            seen_any = True
    if not seen_any:
        ax.text(0.5, 0.5, "No multi-thread pla_only data",
                ha="center", transform=ax.transAxes)
    ax.set_title("IM-A: Parallel Build Overhead (extra segments %)")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Extra segments vs 1-thread (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "parallel_build_overhead.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 10: Retrain impact (DW-A/B) ─────────────────────────────────────────
def plot_retrain_impact(rows: List[Dict], out_dir: Path):
    """DW-A/B: retrain_ms and p99 latency vs epsilon and workload."""
    dynamic = [r for r in rows if r["scenario"] == "dynamic" and r["build_ms"] > 0]
    if not dynamic:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: retrain total time vs epsilon
    ax = axes[0]
    for pla, prows in group_by(dynamic, "pla").items():
        pts = sorted((r["epsilon"], r.get("retrain_ms", 0))
                     for r in prows if r.get("retrain_ms", 0) > 0)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="s", label=pla, color=pla_color(pla))
    ax.set_title("DW-A/B: ε vs Total Retrain Time")
    ax.set_xlabel("ε (epsilon)")
    ax.set_ylabel("retrain_ms (cumulative)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: p99 lookup latency vs epsilon, coloured by workload
    ax = axes[1]
    for pla, prows in group_by(dynamic, "pla").items():
        pts = sorted((r["epsilon"], r["p99_ns"])
                     for r in prows if r["p99_ns"] > 0)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="D", label=pla, color=pla_color(pla))
    ax.set_title("DW-B: ε vs p99 Lookup Latency")
    ax.set_xlabel("ε (epsilon)")
    ax.set_ylabel("p99 latency (ns)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "retrain_impact.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 11: On-disk Rp (pages/query) vs epsilon ─────────────────────────────
def plot_ondisk_rp(rows: List[Dict], out_dir: Path):
    """OD-A/B: io_pages_mean (Rp) vs epsilon per PLA."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk" and r.get("io_pages_mean", 0)]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes,
                                  ["io_pages_mean", "ops_s"],
                                  ["Mean pages/query (Rp)", "Throughput (ops/s)"]):
        for pla, prows in group_by(ondisk, "pla").items():
            pts = sorted((r["epsilon"], float(r.get(metric, 0)))
                         for r in prows if float(r.get(metric, 0)) > 0
                         and r.get("granularity", "item") == "item"
                         and not str(r.get("page_align", "false")).lower() in ("true", "1"))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=pla, color=pla_color(pla))
        ax.set_title(f"OD-A/B: ε vs {label}")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "ondisk_rp.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 12: G3 page-align benefit (OD-E) ────────────────────────────────────
def plot_page_align_benefit(rows: List[Dict], out_dir: Path):
    """OD-E: compare io_pages_mean and ops_s with page-align ON vs OFF."""
    ondisk = [r for r in rows
              if r["scenario"] == "ondisk" and r.get("page_align") is not None]
    if not ondisk:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes,
                                  ["io_pages_mean", "ops_s"],
                                  ["Mean pages/query", "Throughput (ops/s)"]):
        for pla, prows in group_by(ondisk, "pla").items():
            for align in [False, True]:
                align_str = "true" if align else "false"
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
                            label=lbl, color=pla_color(pla))
        ax.set_title(f"OD-E: G3 Page-Align Effect ({label})")
        ax.set_xlabel("ε (epsilon)")
        ax.set_ylabel(label)
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
    out_dir  = Path(args.output)
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
    # Additional plots for new experiments
    plot_routing_comparison(rows, out_dir)
    plot_seg_len_distribution(rows, out_dir)
    plot_parallel_build_overhead(rows, out_dir)
    plot_retrain_impact(rows, out_dir)
    plot_ondisk_rp(rows, out_dir)
    plot_page_align_benefit(rows, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
