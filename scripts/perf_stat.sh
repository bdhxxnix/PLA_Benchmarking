#!/usr/bin/env bash
# scripts/perf_stat.sh
# 封装 perf stat，采集硬件性能计数器并将结果追加到 results/raw/perf.jsonl。
#
# 用法:
#   bash scripts/perf_stat.sh --cmd "build/lookup_bench --algo optimal --epsilon 64"
#   bash scripts/perf_stat.sh --cmd "..." --exp-id my_exp --output results/raw/perf.jsonl
#
# 采集指标: cache-misses, cache-references, branches, branch-misses,
#            instructions, cycles, task-clock

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=""
EXP_ID="perf"
OUTPUT="${REPO_ROOT}/results/raw/perf.jsonl"
REPEATS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cmd)       CMD="$2";    shift 2 ;;
        --exp-id)    EXP_ID="$2"; shift 2 ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        --repeats)   REPEATS="$2";shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CMD" ]]; then
    echo "Usage: $0 --cmd <benchmark_command> [--exp-id <id>] [--output <file>]"
    exit 1
fi

if ! command -v perf &>/dev/null; then
    echo "[perf_stat] perf not found; skipping hardware counters."
    exit 0
fi

EVENTS="cache-misses,cache-references,branches,branch-misses,instructions,cycles,task-clock"

echo "[perf_stat] Running ($REPEATS repeats): $CMD"

# Accumulate totals across repeats.
declare -A totals
totals[cache_misses]=0
totals[cache_references]=0
totals[branches]=0
totals[branch_misses]=0
totals[instructions]=0
totals[cycles]=0
totals[task_clock_ms]=0

for rep in $(seq 1 "$REPEATS"); do
    TMPFILE=$(mktemp /tmp/perf_stat_XXXXXX.txt)
    # perf stat writes metrics to stderr.
    perf stat -e "$EVENTS" -- bash -c "$CMD" 2>"$TMPFILE" || true

    # Parse perf output (format: "  12345  event-name  ...")
    while IFS= read -r line; do
        val=$(echo "$line" | awk '{gsub(/,/,"",$1); print $1+0}')
        case "$line" in
            *cache-misses*)       totals[cache_misses]=$(( ${totals[cache_misses]}     + val )) ;;
            *cache-references*)   totals[cache_references]=$(( ${totals[cache_references]} + val )) ;;
            *branch-misses*)      totals[branch_misses]=$(( ${totals[branch_misses]}   + val )) ;;
            *branches*)           totals[branches]=$(( ${totals[branches]}             + val )) ;;
            *instructions*)       totals[instructions]=$(( ${totals[instructions]}     + val )) ;;
            *cycles*)             totals[cycles]=$(( ${totals[cycles]}                 + val )) ;;
            *task-clock*)         totals[task_clock_ms]=$(echo "${totals[task_clock_ms]} $val" | awk '{printf "%d", $1+$2}') ;;
        esac
    done < "$TMPFILE"
    rm -f "$TMPFILE"
done

# Average over repeats.
avg() { echo "$1 $REPEATS" | awk '{printf "%d", $1/$2}'; }

mkdir -p "$(dirname "$OUTPUT")"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat >> "$OUTPUT" <<JSON
{"exp_id":"${EXP_ID}","timestamp":"${TIMESTAMP}","cmd":"${CMD}","repeats":${REPEATS},"cache_misses":$(avg "${totals[cache_misses]}"),"cache_references":$(avg "${totals[cache_references]}"),"branches":$(avg "${totals[branches]}"),"branch_misses":$(avg "${totals[branch_misses]}"),"instructions":$(avg "${totals[instructions]}"),"cycles":$(avg "${totals[cycles]}"),"task_clock_ms":$(avg "${totals[task_clock_ms]}")}
JSON

echo "[perf_stat] Results appended to $OUTPUT"
