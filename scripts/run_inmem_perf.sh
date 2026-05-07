#!/usr/bin/env bash
# Run comprehensive in-memory benchmarks with HW perf counters.
# Produces: results/raw/inmem_perf.jsonl
#
# PLA selection is via --algo flag (runtime), NOT compile-time.
# All three algorithms are compiled into every binary.
#
# Matrix:
#   scenario:  pla_only, inmem
#   pla:       optimal, swing, greedy       (--algo flag)
#   epsilon:   8, 16, 32, 64, 128, 256, 512, 1024
#   routing:   pgm-index, fiting-tree      (--index flag, inmem only)
#   dataset:   synth uniform 10M, synth lognormal 10M
#   threads:   1
set -euo pipefail

OUTFILE="results/raw/inmem_perf.jsonl"
N_KEYS=10000000
QUERIES=1000000
EPSILONS=(8 16 32 64 128 256 512 1024)
PLAS=(optimal swing greedy)
DISTS=(uniform lognormal)

# Single cmake build (PLA_ALGO doesn't matter since --algo selects at runtime)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPLA_ALGO=optimal > /dev/null 2>&1
cmake --build build -j$(nproc) --target pla_build_bench lookup_bench 2>&1 | tail -1

# Clean output file
:> "$OUTFILE"

for PLA in "${PLAS[@]}"; do
    for DIST in "${DISTS[@]}"; do
        for EPS in "${EPSILONS[@]}"; do
            EXP_ID="IM_${PLA}_e${EPS}_${DIST}"

            # ── pla_only ──────────────────────────────────────────────────
            echo "  pla_only $PLA ε=$EPS $DIST"
            ./build/pla_build_bench \
                --algo "$PLA" --epsilon "$EPS" \
                --dist "$DIST" --n "$N_KEYS" --threads 1 --no-verify \
                --exp-id "${EXP_ID}_pla" \
                >> "$OUTFILE" 2>/dev/null

            # ── inmem: FITing-Tree ────────────────────────────────────────
            echo "  inmem/fit $PLA ε=$EPS $DIST"
            ./build/lookup_bench \
                --algo "$PLA" --epsilon "$EPS" \
                --dist "$DIST" --n "$N_KEYS" --queries "$QUERIES" --threads 1 \
                --index fiting-tree --workload readonly \
                --exp-id "${EXP_ID}_fit" \
                >> "$OUTFILE" 2>/dev/null

            # ── inmem: PGM-index ──────────────────────────────────────────
            echo "  inmem/pgm $PLA ε=$EPS $DIST"
            ./build/lookup_bench \
                --algo "$PLA" --epsilon "$EPS" \
                --dist "$DIST" --n "$N_KEYS" --queries "$QUERIES" --threads 1 \
                --index pgm-index --workload readonly \
                --exp-id "${EXP_ID}_pgm" \
                >> "$OUTFILE" 2>/dev/null
        done
    done
done

echo "Done. Results in $OUTFILE ($(wc -l < "$OUTFILE") rows)"
