#!/usr/bin/env bash
# run_experiments.sh — Configurable experiment runner for pla-learned-index-bench
#
# Usage:
#   ./run_experiments.sh                          # run all experiments, all PLAs
#   ./run_experiments.sh --experiments IM-A,IM-B  # only IM-A and IM-B
#   ./run_experiments.sh --experiments OD-C --plas swing,greedy
#   ./run_experiments.sh --n 1000000 --queries 1000000  # full scale
#   ./run_experiments.sh --dry-run                 # print commands, don't execute
#   ./run_experiments.sh --list                    # list all experiment codes
#
# Experiment codes:
#   IM-A   PLA-only iso-epsilon baseline
#   IM-B   In-memory routing comparison (PGM-index vs FITing-Tree)
#   DW-A   Dynamic retrain cost (write-heavy, 90% inserts)
#   DW-B   Dynamic workload sweep (readonly/balanced/write_heavy)
#   OD-A   On-disk iso-epsilon + iso-RP
#   OD-B   On-disk fixed G1/G2/G3 + G4 compression
#   OD-C   On-disk G1 fetch strategy × PLA
#   OD-D   On-disk G2 granularity × PLA
#   OD-E   On-disk G3 page-alignment × PLA
#   OD-F   On-disk hybrid update workload

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO/build"
RESULTS_DIR="$REPO/results/raw"
N_KEYS=100000
QUERIES=100000
DIST_UNIFORM="uniform"
DIST_LOGNORMAL="lognormal"
DATASET="$REPO/data/sosd_fb_1M_sorted"      # for on-disk benchmarks
DYN_DATASET="$REPO/data/sosd_fb_1M"          # for dynamic benchmarks
THREADS=1
EXPERIMENTS="all"
PLAS="all"
DRY_RUN=false
NO_BUILD=false
COLD_CACHE=false
EPS_IMA="32 64 128 256 512"
EPS_IMB="32 64 128 256"
EPS_DWA="32 64 128 256 512"
EPS_DWB="32 64 128"
EPS_ODA="16 64 128 256 512 1024"
EPS_ODB="16 64 128 512"
EPS_ODC="64 256 1024"
EPS_ODD="4 32 128 256"
EPS_ODE="32 128 512"
EPS_ODF="64 128 256"

# ─── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --plas)        PLAS="$2";        shift 2 ;;
    --n)           N_KEYS="$2";      shift 2 ;;
    --queries)     QUERIES="$2";     shift 2 ;;
    --dataset)     DATASET="$2";     shift 2 ;;
    --dyn-dataset) DYN_DATASET="$2"; shift 2 ;;
    --threads)     THREADS="$2";     shift 2 ;;
    --output-dir)  RESULTS_DIR="$2"; shift 2 ;;
    --eps-ima)     EPS_IMA="$2";     shift 2 ;;
    --eps-imb)     EPS_IMB="$2";     shift 2 ;;
    --eps-dwa)     EPS_DWA="$2";     shift 2 ;;
    --eps-dwb)     EPS_DWB="$2";     shift 2 ;;
    --eps-oda)     EPS_ODA="$2";     shift 2 ;;
    --eps-odb)     EPS_ODB="$2";     shift 2 ;;
    --eps-odc)     EPS_ODC="$2";     shift 2 ;;
    --eps-odd)     EPS_ODD="$2";     shift 2 ;;
    --eps-ode)     EPS_ODE="$2";     shift 2 ;;
    --eps-odf)     EPS_ODF="$2";     shift 2 ;;
    --dry-run)     DRY_RUN=true;     shift ;;
    --no-build)    NO_BUILD=true;     shift ;;
    --cold-cache)  COLD_CACHE=true;   shift ;;
    --list)
      echo "Experiment codes:"
      echo "  IM-A   PLA-only iso-epsilon baseline"
      echo "  IM-B   In-memory routing (PGM-index vs FITing-Tree)"
      echo "  DW-A   Dynamic retrain cost (write-heavy)"
      echo "  DW-B   Dynamic workload sweep"
      echo "  OD-A   On-disk iso-epsilon + iso-RP"
      echo "  OD-B   On-disk fixed G1/G2/G3 + G4 compression"
      echo "  OD-C   On-disk G1 fetch strategy × PLA"
      echo "  OD-D   On-disk G2 granularity × PLA"
      echo "  OD-E   On-disk G3 page-alignment × PLA"
      echo "  OD-F   On-disk hybrid update workload"
      echo "  all    Run all experiments"
      exit 0 ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo ""
      echo "Experiment selection:"
      echo "  --experiments IM-A,IM-B,...   Comma-separated list or 'all' (default: all)"
      echo "  --plas optimal,swing,greedy   Comma-separated list or 'all' (default: all)"
      echo "  --list                        List all experiment codes"
      echo ""
      echo "Data parameters:"
      echo "  --n N              Number of keys (default: 100000)"
      echo "  --queries Q        Number of queries (default: 100000)"
      echo "  --dataset PATH     On-disk dataset path (default: data/sosd_fb_1M_sorted)"
      echo "  --dyn-dataset PATH Dynamic dataset path (default: data/sosd_fb_1M)"
      echo "  --threads T        Thread count (default: 1)"
      echo ""
      echo "Epsilon overrides (space-separated):"
      echo "  --eps-ima \"8 32 128 512\"    Override IM-A epsilons"
      echo "  --eps-imb \"...\"             Override IM-B epsilons"
      echo "  ... (same pattern for --eps-dwa through --eps-odf)"
      echo ""
      echo "Execution control:"
      echo "  --dry-run          Print commands without executing"
      echo "  --no-build         Skip cmake rebuilds"
      echo "  --cold-cache       Run drop_caches.sh before each on-disk run (needs sudo)"
      echo "  --output-dir DIR   Results directory (default: results/raw)"
      exit 0 ;;
    *) echo "Unknown: $1 (use --help)"; exit 1 ;;
  esac
done

# ─── Resolve experiment list ────────────────────────────────────────────────────
ALL_EXPS=(IM-A IM-B DW-A DW-B OD-A OD-B OD-C OD-D OD-E OD-F)
if [[ "$EXPERIMENTS" == "all" ]]; then
  EXPS=("${ALL_EXPS[@]}")
else
  IFS=',' read -ra EXPS <<< "$EXPERIMENTS"
fi

# Resolve PLA list
ALL_PLAS=(optimal swing greedy)
if [[ "$PLAS" == "all" ]]; then
  PLA_LIST=("${ALL_PLAS[@]}")
else
  IFS=',' read -ra PLA_LIST <<< "$PLAS"
fi

# ─── Helpers ─────────────────────────────────────────────────────────────────────
run_cmd() {
  if $DRY_RUN; then
    echo "  [dry-run] $*"
  else
    "$@"
  fi
}

cold_cache() {
  if $COLD_CACHE; then
    sudo bash "$REPO/scripts/drop_caches.sh" 2>/dev/null || true
  fi
}

build_pla() {
  local pla="$1"
  if $NO_BUILD; then
    echo "  [skip build] PLA_ALGO=$pla"
    return 0
  fi
  echo "  [build] PLA_ALGO=$pla"
  run_cmd cmake -S "$REPO" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release -DPLA_ALGO="$pla" -DBUILD_TESTS=OFF > /dev/null 2>&1
  run_cmd cmake --build "$BUILD_DIR" -j"$(nproc)" > /dev/null 2>&1
}

# ─── Main ────────────────────────────────────────────────────────────────────────
echo "=============================================="
echo " pla-learned-index-bench experiment runner"
echo "=============================================="
echo " Experiments: ${EXPS[*]}"
echo " PLAs:        ${PLA_LIST[*]}"
echo " N keys:      $N_KEYS"
echo " N queries:   $QUERIES"
echo " Threads:     $THREADS"
echo " Dataset:     $DATASET"
echo " Dry run:     $DRY_RUN"
echo " Cold cache:  $COLD_CACHE"
echo " Results:     $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

run_experiments() {
  local -n _exps="$1"  # array of experiment codes active in this PLA iteration

  for exp in "${_exps[@]}"; do
    echo ""
    echo "─── $exp ───"

    case "$exp" in

      # ── IM-A: PLA-only iso-epsilon ──────────────────────────────────────────
      IM-A)
        for eps in $EPS_IMA; do
          for dist in $DIST_UNIFORM $DIST_LOGNORMAL; do
            local eid="IMA_${PLA}_e${eps}_${dist}"
            cold_cache
            run_cmd "$BUILD_DIR/pla_build_bench" \
              --algo "$PLA" --epsilon "$eps" --dist "$dist" \
              --n "$N_KEYS" --threads "$THREADS" --exp-id "$eid" \
              >> "$RESULTS_DIR/pla_only.jsonl"
            echo "    OK: $eid"
          done
        done
        ;;

      # ── IM-B: In-memory routing ─────────────────────────────────────────────
      IM-B)
        for eps in $EPS_IMB; do
          for idx in fiting-tree pgm-index; do
            for wl in readonly balanced zipf; do
              local eid="IMB_${PLA}_e${eps}_${idx}_${wl}"
              run_cmd "$BUILD_DIR/lookup_bench" \
                --algo "$PLA" --epsilon "$eps" --dist uniform \
                --n "$N_KEYS" --queries "$QUERIES" --threads "$THREADS" \
                --index "$idx" --workload "$wl" --exp-id "$eid" \
                >> "$RESULTS_DIR/inmem.jsonl"
              echo "    OK: $eid"
            done
          done
        done
        ;;

      # ── DW-A: Dynamic retrain cost ──────────────────────────────────────────
      DW-A)
        for eps in $EPS_DWA; do
          local eid="DWA_${PLA}_e${eps}"
          run_cmd "$BUILD_DIR/dynamic_bench" \
            --algo "$PLA" --epsilon "$eps" --workload write_heavy \
            --n "$N_KEYS" --exp-id "$eid" \
            >> "$RESULTS_DIR/dynamic.jsonl"
          echo "    OK: $eid"
        done
        ;;

      # ── DW-B: Dynamic workload sweep ────────────────────────────────────────
      DW-B)
        for eps in $EPS_DWB; do
          for wl in readonly balanced write_heavy; do
            local eid="DWB_${PLA}_e${eps}_${wl}"
            run_cmd "$BUILD_DIR/dynamic_bench" \
              --algo "$PLA" --epsilon "$eps" --workload "$wl" \
              --n "$N_KEYS" --exp-id "$eid" \
              >> "$RESULTS_DIR/dynamic.jsonl"
            echo "    OK: $eid"
          done
        done
        ;;

      # ── OD-A: On-disk iso-epsilon ───────────────────────────────────────────
      OD-A)
        for eps in $EPS_ODA; do
          local eid="ODA_${PLA}_e${eps}"
          cold_cache
          run_cmd "$BUILD_DIR/ondisk_bench" \
            --algo "$PLA" --epsilon "$eps" --fetch-strategy 1 \
            --dataset "$DATASET" --n "$N_KEYS" --queries "$QUERIES" \
            --exp-id "$eid" >> "$RESULTS_DIR/ondisk.jsonl"
          echo "    OK: $eid"
        done
        ;;

      # ── OD-B: Fixed G1/G2/G3 + G4 compression ──────────────────────────────
      OD-B)
        for eps in $EPS_ODB; do
          # Without compression
          local eid="ODB_${PLA}_e${eps}"
          cold_cache
          run_cmd "$BUILD_DIR/ondisk_bench" \
            --algo "$PLA" --epsilon "$eps" --fetch-strategy 1 \
            --dataset "$DATASET" --n "$N_KEYS" --queries "$QUERIES" \
            --exp-id "$eid" >> "$RESULTS_DIR/ondisk.jsonl"
          echo "    OK: $eid"
          # With G4 compression
          local eid_c="ODB_${PLA}_e${eps}_comp"
          cold_cache
          run_cmd "$BUILD_DIR/ondisk_bench" \
            --algo "$PLA" --epsilon "$eps" --fetch-strategy 1 --compress \
            --dataset "$DATASET" --n "$N_KEYS" --queries "$QUERIES" \
            --exp-id "$eid_c" >> "$RESULTS_DIR/ondisk.jsonl"
          echo "    OK: $eid_c"
        done
        ;;

      # ── OD-C: G1 Fetch strategy × PLA ──────────────────────────────────────
      OD-C)
        for eps in $EPS_ODC; do
          for fs in 0 1 2 3; do
            local eid="ODC_${PLA}_e${eps}_fs${fs}"
            cold_cache
            run_cmd "$BUILD_DIR/ondisk_bench" \
              --algo "$PLA" --epsilon "$eps" --fetch-strategy "$fs" \
              --dataset "$DATASET" --n "$N_KEYS" --queries "$QUERIES" \
              --exp-id "$eid" >> "$RESULTS_DIR/ondisk.jsonl"
            echo "    OK: $eid"
          done
        done
        ;;

      # ── OD-D: G2 Granularity × PLA ─────────────────────────────────────────
      OD-D)
        for eps in $EPS_ODD; do
          for gran in item page; do
            local eid="ODD_${PLA}_e${eps}_${gran}"
            cold_cache
            run_cmd "$BUILD_DIR/ondisk_bench" \
              --algo "$PLA" --epsilon "$eps" --granularity "$gran" \
              --fetch-strategy 1 --dataset "$DATASET" \
              --n "$N_KEYS" --queries "$QUERIES" \
              --exp-id "$eid" >> "$RESULTS_DIR/ondisk.jsonl"
            echo "    OK: $eid"
          done
        done
        ;;

      # ── OD-E: G3 Page-alignment × PLA ──────────────────────────────────────
      OD-E)
        for eps in $EPS_ODE; do
          # Without page-align
          local eid="ODE_${PLA}_e${eps}_noalign"
          cold_cache
          run_cmd "$BUILD_DIR/ondisk_bench" \
            --algo "$PLA" --epsilon "$eps" --fetch-strategy 1 \
            --dataset "$DATASET" --n "$N_KEYS" --queries "$QUERIES" \
            --exp-id "$eid" >> "$RESULTS_DIR/ondisk.jsonl"
          echo "    OK: $eid"
          # With page-align
          local eid_a="ODE_${PLA}_e${eps}_align"
          cold_cache
          run_cmd "$BUILD_DIR/ondisk_bench" \
            --algo "$PLA" --epsilon "$eps" --fetch-strategy 1 --page-align \
            --dataset "$DATASET" --n "$N_KEYS" --queries "$QUERIES" \
            --exp-id "$eid_a" >> "$RESULTS_DIR/ondisk.jsonl"
          echo "    OK: $eid_a"
        done
        ;;

      # ── OD-F: Hybrid update workload ───────────────────────────────────────
      OD-F)
        for eps in $EPS_ODF; do
          for wl in readonly insert hybrid; do
            local eid="ODF_${PLA}_e${eps}_${wl}"
            cold_cache
            run_cmd "$BUILD_DIR/ondisk_bench" \
              --algo "$PLA" --epsilon "$eps" --workload "$wl" \
              --fetch-strategy 1 --dataset "$DATASET" \
              --n "$N_KEYS" --queries "$QUERIES" \
              --exp-id "$eid" >> "$RESULTS_DIR/ondisk.jsonl"
            echo "    OK: $eid"
          done
        done
        ;;

      *)
        echo "  Unknown experiment: $exp (use --list to see codes)"
        ;;
    esac
  done
}

# ── Run per PLA ──────────────────────────────────────────────────────────────────
for PLA in "${PLA_LIST[@]}"; do
  echo ""
  echo "══════════════════════════════════════════"
  echo "  PLA: $PLA"
  echo "══════════════════════════════════════════"

  if ! build_pla "$PLA"; then
    echo "  [ERROR] Build failed for $PLA, skipping"
    continue
  fi

  run_experiments EXPS
done

# ── Summary ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Done."
echo " Results: $RESULTS_DIR"
for f in "$RESULTS_DIR"/*.jsonl; do
  [[ -f "$f" ]] && echo "   $(basename "$f"): $(wc -l < "$f") rows"
done
echo ""
echo " Aggregate: python3 tools/viz/aggregate.py $RESULTS_DIR/ results/agg/results.csv"
echo " Plots:     python3 tools/viz/plots.py results/agg/results.csv results/agg/"
echo "=============================================="
