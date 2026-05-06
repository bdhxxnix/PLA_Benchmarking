#!/usr/bin/env bash
# run_experiments.sh — Configurable experiment runner for pla-learned-index-bench
#
# Quick examples:
#   ./run_experiments.sh                                    # smoke scale, all experiments
#   ./run_experiments.sh --scale full                       # full datasets (200M+)
#   ./run_experiments.sh --scale medium --experiments IM-A,IM-B
#   ./run_experiments.sh --experiments OD-C --plas swing,greedy
#   ./run_experiments.sh --dry-run                          # print commands only
#   ./run_experiments.sh --list                             # list experiment codes
#
# Scale presets (--scale):
#   smoke   100K keys, 100K queries — quick verification (default)
#   medium  1M keys, 1M queries — representative
#   full    auto-detect from dataset, 10M queries — production
#
# Full-scale examples:
#   ./run_experiments.sh --scale full --dataset ~/Projects/Datasets/SOSD/fb_200M_uint64
#   ./run_experiments.sh --scale full --dataset ~/Projects/Datasets/SOSD/osm_800M_uint64_unique \
#       --experiments OD-A,OD-B --plas optimal

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO/build"
RESULTS_DIR="$REPO/results/raw"

SCALE="smoke"
N_KEYS=100000
N_KEYS_DYN=0        # 0 = follow N_KEYS; set explicitly to cap dynamic experiment key count
QUERIES=100000

DATASET="$REPO/data/sosd_fb_1M_sorted"
STRIP_HEADER=true          # auto-strip SOSD header for on-disk use

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

# ─── Helpers ─────────────────────────────────────────────────────────────────────
filesize_to_n() {
  # Return number of uint64 keys in a binary file.
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  stat -c%s "$f" 2>/dev/null | awk '{print int($1 / 8)}'
}

prepare_ondisk_dataset() {
  # If the dataset has a leading value that breaks monotonicity (SOSD quirk),
  # create a header-stripped copy in /dev/shm for fast mmap access.
  local src="$1"
  local dst="$2"

  if [[ -f "$dst" ]] && [[ "$(filesize_to_n "$dst")" -gt 0 ]]; then
    echo "$dst"
    return
  fi

  # Check if the first two values are out of order (SOSD header).
  local first_two
  first_two=$(od -A n -t u8 -N 16 "$src" 2>/dev/null | awk '{print $1, $2}')
  local v1 v2
  v1=$(echo "$first_two" | awk '{print $1}')
  v2=$(echo "$first_two" | awk '{print $2}')

  if [[ "$v1" -gt "$v2" ]] && [[ "$v2" -gt 0 ]]; then
    echo "  [strip-header] Removing unsorted leading value from $(basename "$src")" >&2
    mkdir -p "$(dirname "$dst")"
    dd if="$src" of="$dst" bs=8M iflag=skip_bytes skip=8 status=none 2>/dev/null
    echo "$dst"
  else
    # Dataset is already sorted — use directly.
    echo "$src"
  fi
}

# ─── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)
      SCALE="$2"
      case "$SCALE" in
        smoke)
          N_KEYS=100000;   QUERIES=100000;  N_KEYS_DYN=100000 ;;
        medium)
          N_KEYS=1000000;  QUERIES=1000000; N_KEYS_DYN=1000000 ;;
        full)
          N_KEYS=0;        QUERIES=10000000; N_KEYS_DYN=1000000 ;;  # ondisk uses full; dynamic capped at 1M (NaiveDynamic is O(n²) insert)
        *) echo "Unknown scale: $SCALE (use smoke|medium|full)"; exit 1 ;;
      esac
      shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --plas)        PLAS="$2";        shift 2 ;;
    --n)           N_KEYS="$2";      shift 2 ;;
    --dyn-n)       N_KEYS_DYN="$2";  shift 2 ;;
    --queries)     QUERIES="$2";     shift 2 ;;
    --dataset)     DATASET="$2";      shift 2 ;;
    --no-strip)    STRIP_HEADER=false; shift ;;
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
      echo "=== Scale presets ==="
      echo "  --scale smoke|medium|full   (default: smoke)"
      echo "    smoke   100K keys, 100K queries"
      echo "    medium  1M keys, 1M queries"
      echo "    full    auto-detect from dataset, 10M queries"
      echo ""
      echo "=== Dataset paths ==="
      echo "  --dataset PATH     Dataset for all benchmarks (on-disk, dynamic, in-memory)"
      echo "  --no-strip         Don't auto-strip SOSD header for on-disk"
      echo ""
      echo "  Defaults (smoke/medium): data/sosd_fb_1M_sorted"
      echo "  For --scale full, point at real datasets:"
      echo "    --dataset ~/Projects/Datasets/SOSD/fb_200M_uint64"
      echo "    --dataset ~/Projects/Datasets/SOSD/osm_800M_uint64_unique"
      echo "  The script auto-detects and strips the SOSD header for on-disk use."
      echo ""
      echo "=== Experiment selection ==="
      echo "  --experiments IM-A,IM-B,...  Comma-separated or 'all' (default: all)"
      echo "  --plas optimal,swing,greedy  Comma-separated or 'all' (default: all)"
      echo "  --list                       List all experiment codes"
      echo ""
      echo "=== Data parameters ==="
      echo "  --n N              Override number of keys (0=auto from dataset)"
      echo "  --queries Q        Number of queries"
      echo "  --threads T        Thread count (default: 1)"
      echo ""
      echo "=== Epsilon overrides (space-separated) ==="
      echo "  --eps-ima \"...\" --eps-imb \"...\" --eps-dwa \"...\""
      echo "  --eps-dwb \"...\" --eps-oda \"...\" --eps-odb \"...\""
      echo "  --eps-odc \"...\" --eps-odd \"...\" --eps-ode \"...\" --eps-odf \"...\""
      echo ""
      echo "=== Execution control ==="
      echo "  --dry-run          Print commands without executing"
      echo "  --no-build         Skip cmake rebuilds"
      echo "  --cold-cache       Run drop_caches.sh before each on-disk run (needs sudo)"
      echo "  --output-dir DIR   Results directory (default: results/raw)"
      echo ""
      echo "=== Full-scale examples ==="
      echo "  # Facebook 200M"
      echo "  $0 --scale full --dataset ~/Projects/Datasets/SOSD/fb_200M_uint64"
      echo ""
      echo "  # OSM 800M — on-disk only, optimal PLA"
      echo "  $0 --scale full --dataset ~/Projects/Datasets/SOSD/osm_800M_uint64_unique \\"
      echo "      --experiments OD-A,OD-B,OD-C --plas optimal"
      echo ""
      echo "  # In-memory only"
      echo "  $0 --scale full --dataset ~/Projects/Datasets/SOSD/fb_200M_uint64 \\"
      echo "      --experiments IM-A,IM-B,DW-A --n 200000000"
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
echo " Scale:       $SCALE"
echo " Experiments: ${EXPS[*]}"
echo " PLAs:        ${PLA_LIST[*]}"
echo " N keys:      $N_KEYS  (dynamic: $N_KEYS_DYN)"
echo " N queries:   $QUERIES"
echo " Threads:     $THREADS"
echo " Dataset:     $DATASET"
echo " Dry run:     $DRY_RUN"
echo " Cold cache:  $COLD_CACHE"
echo " Results:     $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# ─── Resolve N_KEYS and prepare dataset (deferred until after header) ───────────
if [[ "$SCALE" == "full" ]] && [[ "$N_KEYS" -eq 0 ]]; then
  echo " Detecting dataset size..."
  N_KEYS=$(filesize_to_n "$DATASET")
  echo " N keys (auto): $N_KEYS"
fi

# N_KEYS_DYN: 0 means "follow N_KEYS" (for smoke/medium); explicit value caps dynamic runs.
if [[ "$N_KEYS_DYN" -eq 0 ]]; then
  N_KEYS_DYN="$N_KEYS"
fi

# Strip SOSD header only when an on-disk experiment is actually being run.
if $STRIP_HEADER && printf '%s\n' "${EXPS[@]}" | grep -q '^OD-'; then
  if [[ -f "$DATASET" ]]; then
    echo " Preparing on-disk dataset..."
    DATASET=$(prepare_ondisk_dataset "$DATASET" "$REPO/data/sosd_ondisk_$(basename "$DATASET")")
    echo " On-disk dataset: $DATASET"
  fi
fi

run_experiments() {
  local -n _exps="$1"

  for exp in "${_exps[@]}"; do
    echo ""
    echo "─── $exp ───"

    case "$exp" in

      # ── IM-A: PLA-only iso-epsilon ──────────────────────────────────────────
      IM-A)
        for eps in $EPS_IMA; do
          if [[ -f "$DATASET" ]]; then
            local eid="IMA_${PLA}_e${eps}_$(basename "$DATASET")"
            run_cmd "$BUILD_DIR/pla_build_bench" \
              --algo "$PLA" --epsilon "$eps" --dataset "$DATASET" \
              --n "$N_KEYS" --threads "$THREADS" --exp-id "$eid" \
              >> "$RESULTS_DIR/pla_only.jsonl"
            echo "    OK: $eid"
          else
            for dist in uniform lognormal; do
              local eid="IMA_${PLA}_e${eps}_${dist}"
              run_cmd "$BUILD_DIR/pla_build_bench" \
                --algo "$PLA" --epsilon "$eps" --dist "$dist" \
                --n "$N_KEYS" --threads "$THREADS" --exp-id "$eid" \
                >> "$RESULTS_DIR/pla_only.jsonl"
              echo "    OK: $eid"
            done
          fi
        done
        ;;

      # ── IM-B: In-memory routing ─────────────────────────────────────────────
      IM-B)
        local dist_or_dataset
        if [[ -f "$DATASET" ]]; then
          dist_or_dataset="--dataset $DATASET"
        else
          dist_or_dataset="--dist uniform"
        fi
        for eps in $EPS_IMB; do
          for idx in fiting-tree pgm-index; do
            for wl in readonly balanced zipf; do
              local eid="IMB_${PLA}_e${eps}_${idx}_${wl}"
              run_cmd "$BUILD_DIR/lookup_bench" \
                --algo "$PLA" --epsilon "$eps" $dist_or_dataset \
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
            --n "$N_KEYS_DYN" --dataset "$DATASET" --exp-id "$eid" \
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
              --n "$N_KEYS_DYN" --dataset "$DATASET" --exp-id "$eid" \
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
          local eid="ODB_${PLA}_e${eps}"
          cold_cache
          run_cmd "$BUILD_DIR/ondisk_bench" \
            --algo "$PLA" --epsilon "$eps" --fetch-strategy 1 \
            --dataset "$DATASET" --n "$N_KEYS" --queries "$QUERIES" \
            --exp-id "$eid" >> "$RESULTS_DIR/ondisk.jsonl"
          echo "    OK: $eid"
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
          local eid="ODE_${PLA}_e${eps}_noalign"
          cold_cache
          run_cmd "$BUILD_DIR/ondisk_bench" \
            --algo "$PLA" --epsilon "$eps" --fetch-strategy 1 \
            --dataset "$DATASET" --n "$N_KEYS" --queries "$QUERIES" \
            --exp-id "$eid" >> "$RESULTS_DIR/ondisk.jsonl"
          echo "    OK: $eid"
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
echo " Post-process:"
echo "   python3 tools/viz/aggregate.py $RESULTS_DIR/ results/agg/results.csv"
echo "   python3 tools/viz/plots.py results/agg/results.csv results/agg/"
echo "=============================================="
