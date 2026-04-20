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

// ─── Perf counter wrapper (Linux perf_event_open) ─────────────────────────────
#ifdef __linux__
#  include <linux/perf_event.h>
#  include <sys/ioctl.h>
#  include <sys/syscall.h>
#  include <unistd.h>

static long perf_event_open_wrap(struct perf_event_attr* hw_event, pid_t pid,
                                  int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct PerfCounters {
    int fd_cache  = -1;
    int fd_instr  = -1;
    int fd_cycles = -1;
    int fd_brmiss = -1;

    void try_open(int& fd, uint32_t type, uint64_t config) {
        struct perf_event_attr pe = {};
        pe.type          = type;
        pe.size          = sizeof(pe);
        pe.config        = config;
        pe.disabled      = 1;
        pe.exclude_kernel= 1;
        pe.exclude_hv    = 1;
        fd = static_cast<int>(perf_event_open_wrap(&pe, 0, -1, -1, 0));
    }

    void open_all() {
        try_open(fd_cache,  PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
        try_open(fd_instr,  PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        try_open(fd_cycles, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
        try_open(fd_brmiss, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    }

    void ctrl(int cmd) {
        if (fd_cache  >= 0) ioctl(fd_cache,  cmd, 0);
        if (fd_instr  >= 0) ioctl(fd_instr,  cmd, 0);
        if (fd_cycles >= 0) ioctl(fd_cycles, cmd, 0);
        if (fd_brmiss >= 0) ioctl(fd_brmiss, cmd, 0);
    }

    void enable()  { ctrl(PERF_EVENT_IOC_ENABLE); }
    void reset()   { ctrl(PERF_EVENT_IOC_RESET);  }
    void disable() { ctrl(PERF_EVENT_IOC_DISABLE);}

    int64_t read_fd(int fd) const {
        if (fd < 0) return 0;
        int64_t val = 0;
        if (::read(fd, &val, sizeof(val)) != static_cast<ssize_t>(sizeof(val)))
            return 0;
        return val;
    }

    int64_t cache_misses()  const { return read_fd(fd_cache);  }
    int64_t instructions()  const { return read_fd(fd_instr);  }
    int64_t cycles()        const { return read_fd(fd_cycles); }
    int64_t branch_misses() const { return read_fd(fd_brmiss); }

    ~PerfCounters() {
        if (fd_cache  >= 0) close(fd_cache);
        if (fd_instr  >= 0) close(fd_instr);
        if (fd_cycles >= 0) close(fd_cycles);
        if (fd_brmiss >= 0) close(fd_brmiss);
    }
};
#else
struct PerfCounters {
    void open_all()  {}
    void enable()    {}
    void reset()     {}
    void disable()   {}
    int64_t cache_misses()  const { return 0; }
    int64_t instructions()  const { return 0; }
    int64_t cycles()        const { return 0; }
    int64_t branch_misses() const { return 0; }
};
#endif

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

// FITing-Tree style: flat segment array + O(log S) binary search for routing.
// This is the "index_lower_bound" approach — equivalent to a sorted array of
// segment breakpoints searched with std::upper_bound.
struct FitingTreeIndex {
    pla::PlaResult pla;

    void build(const std::vector<uint64_t>& keys, int64_t epsilon,
               pla::PlaAlgo algo, pla::PlaOptions opts) {
        pla = pla::build_pla(keys, epsilon, algo, opts);
    }

    pla::SearchRange search_range(uint64_t key) const {
        return pla.search_range(key);
    }

    size_t bytes()       const { return pla.bytes(); }
    int    levels()      const { return 1; }
    size_t seg_cnt_l0()  const { return pla.segments.size(); }
    size_t seg_cnt_l1()  const { return 0; }
    double build_ms()    const { return pla.build_ms; }
    size_t seg_cnt()     const { return pla.segments.size(); }
};

// PGM-index style: 2-level recursive PLA.
// Level-1 narrows the binary search range in level-0 from O(log S) to O(log ε₁).
// ε₁ is chosen as sqrt(S) to balance index size vs search speed.
struct PgmStyleIndex {
    pla::PlaResult level0;  // base PLA over original keys
    pla::PlaResult level1;  // PLA over segment key_lo values
    int64_t        eps0 = 0;
    double         total_build_ms = 0;

    void build(const std::vector<uint64_t>& keys, int64_t epsilon,
               pla::PlaAlgo algo, pla::PlaOptions opts) {
        eps0   = epsilon;
        level0 = pla::build_pla(keys, epsilon, algo, opts);
        total_build_ms = level0.build_ms;

        if (level0.segments.size() > 2) {
            std::vector<uint64_t> seg_keys;
            seg_keys.reserve(level0.segments.size());
            for (const auto& s : level0.segments)
                seg_keys.push_back(s.key_lo);

            // ε₁ = sqrt(S): balances routing accuracy vs level-1 size.
            int64_t eps1 = std::max(int64_t(4),
                static_cast<int64_t>(std::sqrt(static_cast<double>(
                    level0.segments.size()))));
            level1 = pla::build_pla(seg_keys, eps1, algo, opts);
            total_build_ms += level1.build_ms;
        }
    }

    pla::SearchRange search_range(uint64_t key) const {
        const int64_t n = static_cast<int64_t>(level0.n_keys);
        const auto&   segs = level0.segments;

        if (segs.empty()) return {0, n};

        // Fall back to flat search if level-1 is empty (very small datasets).
        if (level1.segments.empty()) {
            return level0.search_range(key);
        }

        // Step 1: level-1 predicts approximate segment index.
        auto sr1   = level1.search_range(key);
        int64_t lo = std::max(int64_t(0), sr1.lo);
        int64_t hi = std::min(sr1.hi, static_cast<int64_t>(segs.size()));

        // Step 2: narrow binary search within [lo, hi) for the segment.
        // Find rightmost segment with key_lo <= key.
        int64_t left = lo, right = hi;
        while (left < right) {
            int64_t mid = left + (right - left) / 2;
            if (segs[static_cast<size_t>(mid)].key_lo <= key) left = mid + 1;
            else right = mid;
        }
        int64_t seg_idx = left - 1;

        if (seg_idx < 0 || seg_idx >= static_cast<int64_t>(segs.size())) {
            // Key is before the first segment.
            return {0, std::min(n, eps0 + 2)};
        }

        // Step 3: predict position using found segment.
        auto pred = static_cast<int64_t>(
            segs[static_cast<size_t>(seg_idx)].predict_raw(key));
        return pla::make_range(pred, eps0, n);
    }

    size_t bytes()      const { return level0.bytes() + level1.bytes(); }
    int    levels()     const { return level1.segments.empty() ? 1 : 2; }
    size_t seg_cnt_l0() const { return level0.segments.size(); }
    size_t seg_cnt_l1() const { return level1.segments.size(); }
    double build_ms()   const { return total_build_ms; }
    size_t seg_cnt()    const { return level0.segments.size(); }
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
        << "\"rss_mb\":0,"
        << "\"fetch_strategy\":-1,\"io_pages\":0"
        << "}\n";
    return 0;
}
