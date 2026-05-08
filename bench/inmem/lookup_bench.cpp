// bench/inmem/lookup_bench.cpp
// End-to-end in-memory lookup benchmark (IM-B experiment).
//
// Index types (--index):
//   fiting-tree  — PLA segments + O(log S) binary search routing (default)
//   pgm-index    — 2-level recursive PLA: level-1 narrows segment search
//
// Both support all three PLA algorithms (optimal/swing/greedy) and the same
// last-mile binary search, so the only variable is the routing layer.
//
// Multi-thread (--threads N): divides queries evenly; measures wall-clock
// throughput. Per-query latency percentiles are only reported for N=1.
//
// Hardware counters: collected via perf_event_open (Linux) when available.
// Falls back to zero-reporting without error.

#include <pla/pla_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::duration<double, std::nano>;

#include "perf_counters.h"
#include "rss.h"

// ─── CLI helpers ─────────────────────────────────────────────────────────────
static std::string get_arg(int argc, char** argv, const char* flag, const char* def = "") {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i+1];
    return def;
}
static bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

// ─── Dataset helpers ─────────────────────────────────────────────────────────
static std::vector<uint64_t> load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }
    auto sz = f.tellg(); f.seekg(0);
    size_t n = sz / sizeof(uint64_t);
    std::vector<uint64_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

static std::vector<uint64_t> gen_uniform_sorted(size_t n) {
    std::mt19937_64 rng(42);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = rng();
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

static std::vector<uint64_t> gen_lognormal_sorted(size_t n) {
    std::mt19937_64 rng(42);
    std::lognormal_distribution<double> dist(0.0, 2.0);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = static_cast<uint64_t>(dist(rng) * 1e9);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// ─── Zipf query generator ────────────────────────────────────────────────────
struct ZipfGen {
    std::vector<size_t> table;
    ZipfGen(size_t n, double alpha, uint64_t seed = 42) {
        table.resize(n);
        std::vector<double> w(n);
        double sum = 0;
        for (size_t i = 0; i < n; ++i) { w[i] = 1.0 / std::pow(i + 1, alpha); sum += w[i]; }
        for (size_t i = 0; i < n; ++i) w[i] /= sum;
        std::vector<double> cdf(n);
        std::partial_sum(w.begin(), w.end(), cdf.begin());
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u(0, 1);
        for (size_t i = 0; i < n; ++i) {
            double r   = u(rng);
            size_t idx = static_cast<size_t>(
                std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
            table[i] = std::min(idx, n - 1);
        }
    }
    size_t operator()(size_t pos) const { return table[pos % table.size()]; }
};

// ─── Binary search within range ──────────────────────────────────────────────
static int64_t range_search(const uint64_t* keys, int64_t lo, int64_t hi,
                             uint64_t target) {
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if      (keys[mid] < target) lo = mid + 1;
        else if (keys[mid] > target) hi = mid;
        else                         return mid;
    }
    return -1;
}

// ─── Percentile helper ───────────────────────────────────────────────────────
static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[std::min(idx, v.size() - 1)];
}

// ─── Index types ─────────────────────────────────────────────────────────────

// ─── Real FITing-Tree: stx::btree (B+-tree) routing over PLA segments ──────
// Imports the real stx::btree from the FITing-Tree submodule to index segments.
// This is the actual FITing-Tree design: PLA segments + B+-tree routing.
// Uses std::greater<KeyType> comparator (descending order) — the convention in
// the real FITing-Tree implementation.

#include <stx/btree.h>

struct FitingTreeIndex {
    // FITing-Tree's stx::btree instantiation (matches fiting_tree.h lines 52-59)
    using BTree = stx::btree<uint64_t,            // key   = segment start key
        std::pair<double, double>,                // data  = {slope, intercept}
        std::pair<uint64_t, std::pair<double, double>>,
        std::greater<uint64_t>,                   // descending key order
        stx::btree_default_map_traits<uint64_t, std::pair<double, double>>,
        false>;

    BTree               routing_tree;
    pla::PlaResult      pla;
    int64_t             epsilon = 64;

    void build(const std::vector<uint64_t>& keys, int64_t eps,
               pla::PlaAlgo algo, pla::PlaOptions opts) {
        epsilon = eps;
        pla     = pla::build_pla(keys, epsilon, algo, opts);

        // Bulk-load segments into the B+-tree in descending key order
        // (matching real FITing-Tree convention: reverse iteration)
        std::vector<std::pair<uint64_t, std::pair<double, double>>> formatted;
        formatted.reserve(pla.segments.size());
        for (auto it = pla.segments.rbegin(); it != pla.segments.rend(); ++it) {
            formatted.emplace_back(it->key_lo,
                std::make_pair(it->slope, static_cast<double>(it->intercept)));
        }
        if (!formatted.empty())
            routing_tree.bulk_load(formatted.begin(), formatted.end());
    }

    pla::SearchRange search_range(uint64_t key) const {
        const int64_t n_signed = static_cast<int64_t>(pla.n_keys);
        if (routing_tree.empty()) return {0, n_signed};

        // lower_bound with std::greater finds first segment with key_lo <= key
        auto it = routing_tree.lower_bound(key);
        if (it == routing_tree.end()) {
            // key is before all segments (smaller than smallest key_lo)
            return {0, std::min(n_signed, epsilon + 2)};
        }

        double slope     = it->second.first;
        double intercept = it->second.second;
        auto   pred      = static_cast<int64_t>(
            slope * static_cast<double>(key - it->first) + intercept);
        return pla::make_range(pred, epsilon, n_signed);
    }

    size_t bytes()       const { return pla.bytes(); }
    int    levels()      const { return 1; }
    size_t seg_cnt_l0()  const { return pla.segments.size(); }
    size_t seg_cnt_l1()  const { return 0; }
    double build_ms()    const { return pla.build_ms; }
    size_t seg_cnt()     const { return pla.segments.size(); }
};

// ─── Real PGM-index: recursive multi-level PLA ──────────────────────────────
// Builds levels recursively (epsilon for level-0, epsilon_recursive=4 for
// upper levels) until the top level has ≤ 1 segment.  This matches the real
// PGM-index algorithm in third_party/PGM-index.
//
// Key difference from our previous 2-level PgmStyleIndex:
//   - Upper levels use a small fixed epsilon (4), not sqrt(S)
//   - Builds as many levels as needed, not just 2
//   - Segment routing traverses ALL levels top-down

struct PgmStyleIndex {
    static constexpr int64_t EPS_RECURSIVE = 4;  // matches PGM-index default

    std::vector<pla::Segment> all_segments;       // all levels concatenated
    std::vector<size_t>       levels_offsets;     // start index of each level
    size_t                    n_keys  = 0;
    int64_t                   eps0    = 0;
    double                    build_ms_total = 0;
    int                       num_levels = 0;

    void build(const std::vector<uint64_t>& keys, int64_t epsilon,
               pla::PlaAlgo algo, pla::PlaOptions opts) {
        eps0   = epsilon;
        n_keys = keys.size();

        // ── Level 0: segments over original keys ──────────────────────────
        pla::PlaResult l0 = pla::build_pla(keys, epsilon, algo, opts);
        build_ms_total = l0.build_ms;
        size_t seg_cnt = l0.segments.size();

        all_segments.reserve(seg_cnt * 2);
        levels_offsets.push_back(0);
        for (auto& s : l0.segments)
            all_segments.push_back(std::move(s));
        levels_offsets.push_back(all_segments.size());

        // ── Upper levels: segments over segment key_lo values ─────────────
        // Uses EPS_RECURSIVE (4) for routing accuracy.
        // Repeats until the level has ≤ 1 segment.
        size_t prev_n = seg_cnt;
        while (prev_n > 1) {
            std::vector<uint64_t> seg_keys;
            seg_keys.reserve(prev_n);
            size_t offset = levels_offsets[levels_offsets.size() - 2];
            for (size_t i = offset; i < offset + prev_n; ++i)
                seg_keys.push_back(all_segments[i].key_lo);

            pla::PlaResult lvl = pla::build_pla(seg_keys, EPS_RECURSIVE, algo, opts);
            build_ms_total += lvl.build_ms;
            prev_n = lvl.segments.size();

            for (auto& s : lvl.segments)
                all_segments.push_back(std::move(s));
            levels_offsets.push_back(all_segments.size());
        }

        num_levels = static_cast<int>(levels_offsets.size()) - 1;
    }

    pla::SearchRange search_range(uint64_t key) const {
        const int64_t n_signed = static_cast<int64_t>(n_keys);
        if (all_segments.empty())
            return {0, n_signed};

        // ── Multi-level routing (top-down) ────────────────────────────────
        // Start from the top-level segment
        int64_t seg_idx = static_cast<int64_t>(
            levels_offsets[levels_offsets.size() - 2]);

        for (int lvl = num_levels - 2; lvl >= 0; --lvl) {
            int64_t lvl_begin = static_cast<int64_t>(levels_offsets[lvl]);
            int64_t lvl_end   = static_cast<int64_t>(levels_offsets[lvl + 1]) - 1;

            const auto& seg  = all_segments[static_cast<size_t>(seg_idx)];
            int64_t    pred  = static_cast<int64_t>(seg.predict_raw(key));
            int64_t    lo    = std::max(lvl_begin, pred - EPS_RECURSIVE - 1);
            int64_t    hi    = std::min(lvl_end,   pred + EPS_RECURSIVE + 2);

            // Find rightmost segment with key_lo <= key within [lo, hi]
            int64_t left = lo, right = hi;
            while (left < right) {
                int64_t mid = left + (right - left) / 2;
                if (all_segments[static_cast<size_t>(mid)].key_lo <= key)
                    left = mid + 1;
                else
                    right = mid;
            }
            seg_idx = std::max(lvl_begin, left - 1);
        }

        // ── Last-mile: use the base-level segment for final prediction ────
        if (seg_idx >= 0 &&
            seg_idx < static_cast<int64_t>(levels_offsets[1])) {
            const auto& seg  = all_segments[static_cast<size_t>(seg_idx)];
            int64_t    pred  = static_cast<int64_t>(seg.predict_raw(key));
            return pla::make_range(pred, eps0, n_signed);
        }
        return {0, n_signed};
    }

    size_t bytes()      const { return all_segments.size() * sizeof(pla::Segment); }
    int    levels()     const { return num_levels; }
    size_t seg_cnt_l0() const {
        return levels_offsets.size() >= 2
            ? levels_offsets[1] - levels_offsets[0] : 0;
    }
    size_t seg_cnt_l1() const {
        // Total of all upper-level segments
        return all_segments.size() - seg_cnt_l0();
    }
    double build_ms()   const { return build_ms_total; }
    size_t seg_cnt()    const { return seg_cnt_l0(); }
};

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int64_t     epsilon    = std::stoll(get_arg(argc, argv, "--epsilon", "64"));
    std::string algo_s     = get_arg(argc, argv, "--algo",     "optimal");
    int         threads    = std::stoi(get_arg(argc, argv, "--threads",  "1"));
    std::string dataset    = get_arg(argc, argv, "--dataset",  "");
    size_t      n_synth    = std::stoull(get_arg(argc, argv, "--n",      "1000000"));
    std::string dist_s     = get_arg(argc, argv, "--dist",     "uniform");
    std::string workload   = get_arg(argc, argv, "--workload", "readonly");
    size_t      n_queries  = std::stoull(get_arg(argc, argv, "--queries","1000000"));
    double      zipf_alpha = std::stod(get_arg(argc, argv,   "--zipf-alpha","1.0"));
    std::string index_type = get_arg(argc, argv, "--index",   "fiting-tree");
    std::string exp_id     = get_arg(argc, argv, "--exp-id",  "inmem");
    bool        verbose    = has_flag(argc, argv, "--verbose");

    // ── Load / generate keys ──────────────────────────────────────────────────
    std::vector<uint64_t> keys;
    std::string ds_name;
    if (!dataset.empty()) {
        keys    = load_binary(dataset);
        ds_name = dataset;
    } else if (dist_s == "lognormal") {
        keys    = gen_lognormal_sorted(n_synth);
        ds_name = "synth_lognormal_" + std::to_string(n_synth);
    } else {
        keys    = gen_uniform_sorted(n_synth);
        ds_name = "synth_uniform_" + std::to_string(n_synth);
    }
    std::sort(keys.begin(), keys.end());
    const size_t n = keys.size();

    size_t rss_before = get_rss_mb();

    // ── Build index ───────────────────────────────────────────────────────────
    pla::PlaAlgo    algo = pla::algo_from_string(algo_s);
    pla::PlaOptions opts;
    opts.threads = static_cast<unsigned>(threads);

    FitingTreeIndex fiting_idx;
    PgmStyleIndex   pgm_idx;
    size_t          seg_cnt_l0, seg_cnt_l1;
    size_t          index_bytes;
    double          build_ms;
    int             index_levels;
    std::string     index_name;

    if (index_type == "pgm-index") {
        pgm_idx.build(keys, epsilon, algo, opts);
        seg_cnt_l0   = pgm_idx.seg_cnt_l0();
        seg_cnt_l1   = pgm_idx.seg_cnt_l1();
        index_bytes  = pgm_idx.bytes();
        build_ms     = pgm_idx.build_ms();
        index_levels = pgm_idx.levels();
        index_name   = "PGM-index";
    } else {
        fiting_idx.build(keys, epsilon, algo, opts);
        seg_cnt_l0   = fiting_idx.seg_cnt_l0();
        seg_cnt_l1   = fiting_idx.seg_cnt_l1();
        index_bytes  = fiting_idx.bytes();
        build_ms     = fiting_idx.build_ms();
        index_levels = fiting_idx.levels();
        index_name   = "FITing-Tree";
    }

    if (verbose)
        std::cerr << index_name << ": " << seg_cnt_l0 << " segs (L0), "
                  << seg_cnt_l1 << " segs (L1), build=" << build_ms << " ms\n";

    size_t rss_after = get_rss_mb();
    int64_t rss_mb = static_cast<int64_t>(rss_after) - static_cast<int64_t>(rss_before);

    // ── Build query set ───────────────────────────────────────────────────────
    std::vector<uint64_t> queries;
    queries.reserve(n_queries);
    std::mt19937_64 rng(123);

    if (workload == "zipf") {
        ZipfGen zgen(n, zipf_alpha);
        for (size_t q = 0; q < n_queries; ++q)
            queries.push_back(keys[zgen(q)]);
    } else if (workload == "balanced") {
        std::uniform_int_distribution<size_t>   idx_dist(0, n - 1);
        std::uniform_int_distribution<uint64_t> miss_dist;
        for (size_t q = 0; q < n_queries; ++q) {
            if (q % 2 == 0) queries.push_back(keys[idx_dist(rng)]);
            else             queries.push_back(miss_dist(rng));
        }
    } else { // readonly
        std::uniform_int_distribution<size_t> idx_dist(0, n - 1);
        for (size_t q = 0; q < n_queries; ++q)
            queries.push_back(keys[idx_dist(rng)]);
    }

    // ── Warm-up (10 % of queries, not counted) ────────────────────────────────
    volatile int64_t sink = 0;
    for (size_t q = 0; q < n_queries / 10; ++q) {
        pla::SearchRange r =
            (index_type == "pgm-index")
                ? pgm_idx.search_range(queries[q])
                : fiting_idx.search_range(queries[q]);
        sink ^= range_search(keys.data(), r.lo, r.hi, queries[q]);
    }
    (void)sink;

    // ── Timed run ─────────────────────────────────────────────────────────────
    std::vector<double> latencies;
    double ops_s    = 0.0;
    double p50 = 0, p95 = 0, p99 = 0;
    int64_t hw_cache = 0, hw_instr = 0, hw_cycles = 0, hw_brmiss = 0;

    if (threads == 1) {
        // Single-thread: per-query latency + hardware counters.
        latencies.reserve(n_queries);

        PerfCounters perf;
        perf.open_all();
        perf.reset();
        perf.enable();

        auto t0 = Clock::now();
        if (index_type == "pgm-index") {
            for (size_t q = 0; q < n_queries; ++q) {
                auto qt0 = Clock::now();
                auto r   = pgm_idx.search_range(queries[q]);
                sink ^= range_search(keys.data(), r.lo, r.hi, queries[q]);
                latencies.push_back(Ns(Clock::now() - qt0).count());
            }
        } else {
            for (size_t q = 0; q < n_queries; ++q) {
                auto qt0 = Clock::now();
                auto r   = fiting_idx.search_range(queries[q]);
                sink ^= range_search(keys.data(), r.lo, r.hi, queries[q]);
                latencies.push_back(Ns(Clock::now() - qt0).count());
            }
        }
        double total_ns = Ns(Clock::now() - t0).count();

        perf.disable();
        hw_cache  = perf.cache_misses();
        hw_instr  = perf.instructions();
        hw_cycles = perf.cycles();
        hw_brmiss = perf.branch_misses();

        ops_s = (n_queries / total_ns) * 1e9;
        p50   = percentile(latencies, 50.0);
        p95   = percentile(latencies, 95.0);
        p99   = percentile(latencies, 99.0);

    } else {
        // Multi-thread: wall-clock throughput only (no per-query latency).
        std::atomic<bool> go{false};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(threads));

        size_t per_thread = n_queries / static_cast<size_t>(threads);

        auto t0 = Clock::now();
        go.store(true, std::memory_order_release);

        for (int t = 0; t < threads; ++t) {
            size_t start = static_cast<size_t>(t) * per_thread;
            size_t count = (t == threads - 1)
                ? n_queries - start : per_thread;

            workers.emplace_back([&, start, count]() {
                volatile int64_t local_sink = 0;
                if (index_type == "pgm-index") {
                    for (size_t q = start; q < start + count; ++q) {
                        auto r = pgm_idx.search_range(queries[q]);
                        local_sink ^= range_search(
                            keys.data(), r.lo, r.hi, queries[q]);
                    }
                } else {
                    for (size_t q = start; q < start + count; ++q) {
                        auto r = fiting_idx.search_range(queries[q]);
                        local_sink ^= range_search(
                            keys.data(), r.lo, r.hi, queries[q]);
                    }
                }
                (void)local_sink;
            });
        }

        for (auto& w : workers) w.join();
        double wall_ns = Ns(Clock::now() - t0).count();
        ops_s = (n_queries / wall_ns) * 1e9;
        // p50/p95/p99 left as 0 for multi-thread.
    }
    (void)sink;

    std::cout << std::fixed << std::setprecision(3)
        << "{"
        << "\"exp_id\":\""       << exp_id       << "\","
        << "\"scenario\":\"inmem\","
        << "\"index\":\""        << index_name   << "\","
        << "\"routing\":\""      << index_type   << "\","
        << "\"pla\":\""          << algo_s       << "\","
        << "\"epsilon\":"        << epsilon      << ","
        << "\"threads\":"        << threads      << ","
        << "\"dataset\":\""      << ds_name      << "\","
        << "\"workload\":\""     << workload     << "\","
        << "\"build_ms\":"       << build_ms     << ","
        << "\"seg_cnt\":"        << seg_cnt_l0   << ","
        << "\"seg_cnt_l1\":"     << seg_cnt_l1   << ","
        << "\"index_levels\":"   << index_levels << ","
        << "\"bytes_index\":"    << index_bytes  << ","
        << "\"ops_s\":"          << ops_s        << ","
        << "\"p50_ns\":"         << p50          << ","
        << "\"p95_ns\":"         << p95          << ","
        << "\"p99_ns\":"         << p99          << ","
        << "\"cache_misses\":"   << hw_cache     << ","
        << "\"branches\":0,"
        << "\"branch_misses\":"  << hw_brmiss    << ","
        << "\"instructions\":"   << hw_instr     << ","
        << "\"cycles\":"         << hw_cycles    << ","
        << "\"rss_mb\":"        << rss_mb       << ","
        << "\"fetch_strategy\":-1,\"io_pages\":0"
        << "}\n";
    return 0;
}
