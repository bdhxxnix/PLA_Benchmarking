# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# First-time setup (Ubuntu 22.04)
bash scripts/bootstrap_ubuntu22.sh
git submodule update --init --recursive
bash scripts/apply_patches.sh

# Configure and build (Release — all three PLA algorithms compiled in)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# PLA is selected at RUNTIME via --algo flag, no rebuild needed
./build/pla_build_bench --algo swing --epsilon 64 --dist uniform --n 1000000
./build/pla_build_bench --algo greedy --epsilon 64 --dist uniform --n 1000000

# Run unit tests
cd build && ctest --output-on-failure

# Optional features
cmake -S . -B build -DUSE_JEMALLOC=ON -DUSE_MKL=ON -DUSE_URCU=ON
```

Build type defaults to `Release` (`-O3 -march=native`). Use `-DCMAKE_BUILD_TYPE=Debug` for `-O0 -g`.

## Running Benchmarks

```bash
# Run a full experiment matrix from a YAML config
python3 tools/runner/run.py configs/exp_example.yaml

# Smoke test (first 4 cases, 100K keys)
python3 tools/runner/run.py configs/exp_example.yaml --smoke

# Smoke test with real SOSD data (uses data/sosd_fb_1M, a 1M subset)
python3 tools/runner/run.py configs/exp_smoke_sosd.yaml

# Dry run (print commands without executing)
python3 tools/runner/run.py configs/exp_example.yaml --dry-run

# Filter by scenario or PLA
python3 tools/runner/run.py configs/exp_example.yaml --filter-scenario inmem --filter-pla optimal

# In-memory perf-counter sweep (all 3 PLAs, 8 epsilons, 2 dists)
bash scripts/run_inmem_perf.sh

# Aggregate JSONL results and generate plots
python3 tools/viz/aggregate.py --input results/raw --output results/agg
python3 tools/viz/plots.py --input results/agg/results.csv --output results/agg

# Flush page cache before on-disk benchmarks (requires root)
sudo bash scripts/drop_caches.sh
```

Direct benchmark invocation examples:
```bash
# PLA selection is at RUNTIME via --algo (all three compiled in)
./build/pla_build_bench --algo swing --epsilon 64 --dist uniform --n 1000000

# With real dataset file (binary uint64, little-endian)
./build/pla_build_bench --epsilon 64 --dataset data/sosd_fb_1M --n 1000000

# In-memory lookup (PGM-index or FITing-Tree)
./build/lookup_bench --algo optimal --epsilon 64 --dist lognormal \
    --n 1000000 --queries 1000000 --index pgm-index

# Dynamic with dataset + workload
./build/dynamic_bench --algo optimal --epsilon 128 --dataset data/sosd_fb_1M \
    --workload write_heavy --n 100000

# Ondisk with G4 compression + O_DIRECT
./build/ondisk_bench --dataset data/sosd_fb_1M_sorted --epsilon 256 \
    --compress --direct-io --fetch-strategy 0 --queries 100000
```

### Preparing SOSD datasets for smoke tests

The full SOSD datasets are too large for quick testing. Extract a tiny subset:

```bash
# Extract first 1M keys from fb_200M_uint64 (~8 MB)
dd if=~/Projects/Datasets/SOSD/fb_200M_uint64 of=data/sosd_fb_1M bs=8 count=1000000
```

The benchmark reads raw little-endian uint64 values. It sorts loaded data, so unsorted inputs are safe.

## Architecture

### PLA Algorithm Layer (`pla/include/pla/`)

The core abstraction is a unified C++ header-only library with a single entry point: `build_pla()` in `pla_api.h`. All three algorithms produce identical `Segment` structs (`key_lo`, `key_hi`, `slope`, `intercept`, `rank_lo`, `rank_hi`) and a `SearchRange` (predicted bounds `[lo, hi)` for binary search).

The algorithm is selected **at runtime** via the `--algo optimal|swing|greedy` CLI flag,
which dispatches through `build_pla()` in `pla_api.h`. All three implementations are
compiled into every binary — no rebuild needed to switch. The compile-time
`-DPLA_ALGO` only sets the default when `--algo` is omitted.

- `alg_optimal.h` — wraps PGM-index's `make_segmentation[_par]` (O(n), provably minimal segments)
- `alg_swing.h` — SwingFilter from FITing-Tree (O(1) per segment, pivot = segment start)
- `alg_greedy.h` — GreedyPLA variant (pivot = midpoint of first two points)

### Benchmark Scenarios (`bench/`)

Four independent executables, each a self-contained scenario:

| Executable | Scenario | Key dependency |
|---|---|---|
| `pla_build_bench` | PLA construction only | pla_lib |
| `lookup_bench` | In-memory point lookup | PGM-index, SOSD |
| `dynamic_bench` | Insert + lookup mixed | LOFT (needs MKL + urcu) |
| `ondisk_bench` | Page-level disk I/O emulation | mmap + posix_fadvise |

`dynamic_bench` and `ondisk_bench` are conditionally built (`BUILD_DYNAMIC`, `BUILD_ONDISK` CMake flags, both ON by default). LOFT benchmarks require `USE_MKL=ON -DUSE_URCU=ON`; a `NaiveDynamic` fallback exists for compile-only testing.

### Experiment Pipeline

```
configs/*.yaml → run.py (cartesian product expansion)
    → benchmark binary per case (PLA selected at runtime via --algo)
    → results/raw/{scenario}.jsonl (JSONL append, one object per run)
    → aggregate.py → results/agg/results.csv
    → plots.py → results/agg/*.png (28 chart types)
```

YAML configs define a matrix of `scenario × pla × epsilon × threads × dataset × workload × fetch_strategy`. All three PLA algorithms are compiled into every binary; the runner passes `--algo` at runtime.

The `data/` directory holds real dataset files (binary uint64 LE). When a config references a dataset name not starting with `synth_`, the runner resolves it as `data/<dataset_name>`. SOSD datasets live in `~/Projects/Datasets/SOSD/`, with tiny subsets extracted into `data/` for smoke testing.

### Experiment Configs

| Config | Focus | Key dimensions |
|---|---|---|
| `exp_example.yaml` | Demo (1M synth) | All 4 scenarios, 3 PLAs, 2 epsilons |
| `exp_smoke_sosd.yaml` | Smoke with real data | pla_only + inmem, `data/sosd_fb_1M` |
| `exp_IMA.yaml` | PLA-only iso-epsilon | ε=8..8192, threads 1/2/4/8, segment stats |
| `exp_IMB.yaml` | In-mem routing | PGM-index vs FITing-Tree, 3 workloads |
| `exp_DWA.yaml` | Dynamic retrain cost | write_heavy (90% ins), 5 ε values |
| `exp_DWB.yaml` | Dynamic workload sweep | readonly/balanced/write_heavy, threads 1/4 |
| `exp_ODA.yaml` | On-disk iso-ε + iso-RP | ε=16..1024, `--target-rp` mode |
| `exp_ODB.yaml` | On-disk fixed G1/G2/G3 (+G4) | ε=16..512, `--compress` for G4 |
| `exp_ODC.yaml` | G1 fetch × PLA | 4 strategies × 3 ε values |
| `exp_ODD.yaml` | G2 granularity × PLA | item vs page, 5 ε values |
| `exp_ODE.yaml` | G3 page-align × PLA | align on/off, 3 ε values |
| `exp_ODF.yaml` | Hybrid workload | readonly/insert/hybrid, delta buffer |
| `fb_experiment.yaml` | Facebook 200M real | pla_only + inmem |
| `exp_inmem_perf.yaml` | In-memory perf counters | pla_only + inmem, ε=8..1024, 2 dists, perf counters |

### Helper Scripts

| Script | Purpose |
|---|---|
| `scripts/run_inmem_perf.sh` | Full inmem perf-counter sweep (144 runs, all PLAs × ε × dists × routings) |
| `scripts/bootstrap_ubuntu22.sh` | One-shot system dependency install |
| `scripts/apply_patches.sh` | Idempotent submodule patch application |
| `scripts/drop_caches.sh` | Flush page cache (requires root, for ondisk benchmarks) |

### Third-Party Submodules (`third_party/`)

- **PGM-index** — provides `internal::make_segmentation` used by OptimalPLA
- **FITing-Tree** — provides SwingFilter reference used by `alg_swing.h`
- **LOFT** — dynamic learned index engine for `dynamic_bench`
- **Efficient-Disk-Learned-Index** — on-disk benchmark reference (emulated via mmap)
- **SOSD** — standard datasets and baseline competitors

Submodules require patches via `scripts/apply_patches.sh` (idempotent). The `adapters/` directory contains the patch files and documents injection points.

### Output Schema

Results are JSONL with fields: `exp_id`, `scenario`, `pla`, `epsilon`, `threads`, `dataset`, `workload`, `build_ms`, `seg_cnt`, `bytes_index`, `ops_s`, `p50_ns`, `p95_ns`, `p99_ns`, `cache_misses`, `instructions`, `cycles`, `branch_misses`, `rss_mb`, `fetch_strategy`, `n_keys`, `dup_runs`. The aggregation step adds `cache_miss_rate` and `IPC` derived columns.

Scenario-specific fields: `routing`/`index_levels`/`seg_cnt_l1` (inmem), `retrain_ms`/`retrain_count`/`retrain_p50_ms` (dynamic), `io_pages_mean`/`io_pages_p50` (ondisk), `compress`/`bytes_compressed`/`direct_io` (ondisk), `seg_len_mean`/`slope_mean`/`slope_std`/`intercept_std` (pla_only).

### Benchmark CLI Flags

| Benchmark | Key flags |
|---|---|
| `pla_build_bench` | `--epsilon`, `--algo`, `--threads`, `--dataset`, `--n`, `--dist uniform\|lognormal`, `--no-verify`, `--exp-id` |
  |  | NOTE: omit `--dataset` for synthetic data; only use `--dataset` for binary files |
| `lookup_bench` | `--epsilon`, `--algo`, `--threads`, `--dataset`, `--n`, `--dist`, `--queries`, `--workload readonly\|balanced\|zipf`, `--index fiting-tree\|pgm-index`, `--exp-id` |
| `dynamic_bench` | `--epsilon`, `--algo`, `--threads`, `--dataset`, `--n`, `--insert-ratio`, `--workload readonly\|balanced\|write_heavy`, `--sample-rate`, `--exp-id` |
| `ondisk_bench` | `--epsilon`, `--algo`, `--dataset`, `--n`, `--queries`, `--fetch-strategy 0\|1\|2\|3`, `--granularity item\|page`, `--page-align`, `--target-rp`, `--workload readonly\|insert\|hybrid`, `--compress`, `--direct-io`, `--exp-id` |

### Shared Bench Headers (`bench/`)

- `perf_counters.h` — Linux `perf_event_open` wrapper (cache-misses, instructions, cycles, branch-misses). Non-Linux: no-op stubs.
- `rss.h` — RSS measurement via `/proc/self/statm`.

All four benchmarks link to `pla_lib` which includes `bench/` in its include path. Use `#include "perf_counters.h"` and `#include "rss.h"` directly.

### Recent Feature Additions

- **Runtime PLA selection** (May 2026): All benchmarks accept `--algo optimal|swing|greedy` at runtime. No cmake rebuild needed to switch algorithms. The `-DPLA_ALGO` cmake flag only sets the default.
- **HW counters** (May 2026): All 4 benchmarks report real `cache_misses`, `instructions`, `cycles`, `branch_misses` via shared `perf_counters.h`. Enabled automatically on Linux; no extra flags needed.
- **RSS measurement**: All benchmarks report `rss_mb` (delta from before to after build) via shared `rss.h`.
- **In-memory perf sweep script**: `scripts/run_inmem_perf.sh` runs 144 benchmark cases covering all PLA × ε × routing × distribution combinations, producing `results/raw/inmem_perf.jsonl`.
- **28 plot types**: plots.py generates 28 chart types including space-time tradeoff curves, perf-counter correlation scatter plots, and per-routing facet grids.
- **dynamic_bench `--dataset`**: Supports loading external binary datasets (sorted uint64 LE), split into initial keys + insert stream.
- **G4 compression (`--compress`)**: Reports `bytes_compressed` (float slope+intercept = 40 bytes/segment vs 48 uncompressed) alongside `bytes_index` and `compress` flag. Both fields are now propagated through aggregate.py.
- **O_DIRECT (`--direct-io`)**: `ondisk_bench` supports `O_DIRECT` I/O via `DirectIOFile` alongside existing mmap `DiskFile`. Keys are loaded into a heap buffer; page touches use `pread`. Reports `direct_io` flag in JSONL.

### Analysis

Comprehensive experiment analysis is in `analysis.md` at the repo root. It covers:
- Space-time tradeoff curves (seg_cnt vs throughput/latency) for PGM-index and FITing-Tree
- Perf-counter evidence chain (cache misses, IPC, branch mispredictions)
- Epsilon sweep across three regimes (segment-dominated, sweet spot, last-mile-dominated)
- Central finding: SwingFilter achieves lowest cache-miss rate and highest throughput despite 25-38% more segments than OptimalPLA
