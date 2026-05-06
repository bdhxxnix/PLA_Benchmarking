// bench/dynamic/dynamic_bench.cpp
// LOFT-style dynamic workload benchmark (DW-A and DW-B experiments).
//
// Falls back to NaiveDynamic (sorted vector + periodic retrain) when LOFT
// is not available.
//
// Workload modes (--workload):
//   readonly    — 0 % inserts  (pure lookup after bulk-load)
//   write_heavy — 90 % inserts (DW-A: model cost under heavy churn)
//   balanced    — 50 % inserts (DW-B default)
//
// Per-operation latency is sampled every --sample-rate ops (default 100).
// Reports p50/p95/p99 for lookup latency and retrain latency separately.

#include <pla/pla_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "perf_counters.h"
#include "rss.h"

using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::duration<double, std::nano>;
using Ms    = std::chrono::duration<double, std::milli>;

// ─── CLI helpers ─────────────────────────────────────────────────────────────
static std::string get_arg(int argc, char** argv, const char* f, const char* d = "") {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], f)) return argv[i + 1];
    return d;
}

// ─── Percentile helper ───────────────────────────────────────────────────────
static double vec_pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[std::min(idx, v.size() - 1)];
}

// ─── Dataset loader ─────────────────────────────────────────────────────────
// max_keys=0 means read all; otherwise read at most max_keys entries.
static std::vector<uint64_t> load_binary(const std::string& path, size_t max_keys = 0) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }
    auto sz = f.tellg(); f.seekg(0);
    size_t n = sz / sizeof(uint64_t);
    if (max_keys > 0 && max_keys < n) n = max_keys;
    std::vector<uint64_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(uint64_t));
    return v;
}

// ─── NaiveDynamic: sorted vector + periodic retrain ──────────────────────────
struct NaiveDynamic {
    std::vector<uint64_t> data;
    pla::PlaResult        index;
    int64_t               epsilon;
    pla::PlaAlgo          algo;
    pla::PlaOptions       opts;

    // Retrain statistics.
    size_t retrain_count  = 0;
    double retrain_ms_total = 0;
    std::vector<double> retrain_times_ms;   // per-retrain duration

    // Latency samples (sampled every sample_rate ops).
    std::vector<double> lookup_lat_ns;      // all lookup samples (ns)
    std::vector<double> lookup_lat_retrain_ns; // lookups during retrain window
    size_t sample_rate = 100;
    size_t op_count    = 0;

    bool in_retrain_window = false;         // set true during retrain

    static constexpr size_t RETRAIN_INTERVAL = 10000; // inserts between retrains

    // Accumulated hardware counters across all builds (bulk + retrains).
    PerfCounters perf;
    int64_t hw_cache  = 0;
    int64_t hw_instr  = 0;
    int64_t hw_cycles = 0;
    int64_t hw_brmiss = 0;

    NaiveDynamic(int64_t eps, pla::PlaAlgo a, pla::PlaOptions o,
                 size_t sr = 100)
        : epsilon(eps), algo(a), opts(o), sample_rate(sr) {
        perf.open_all();
    }

    // Bulk-load initial dataset without triggering retrain.
    void bulk_load(std::vector<uint64_t>& init_keys) {
        data = init_keys;
        perf.reset();
        perf.enable();
        auto t0 = Clock::now();
        index = pla::build_pla(data, epsilon, algo, opts);
        double ms = Ms(Clock::now() - t0).count();
        perf.disable();
        hw_cache  += perf.cache_misses();
        hw_instr  += perf.instructions();
        hw_cycles += perf.cycles();
        hw_brmiss += perf.branch_misses();
        retrain_times_ms.push_back(ms);
        retrain_ms_total += ms;
        ++retrain_count;
    }

    void insert(uint64_t key) {
        auto pos = std::lower_bound(data.begin(), data.end(), key);
        data.insert(pos, key);

        if (data.size() % RETRAIN_INTERVAL == 0) {
            in_retrain_window = true;
            perf.reset();
            perf.enable();
            auto t0 = Clock::now();
            index = pla::build_pla(data, epsilon, algo, opts);
            double ms = Ms(Clock::now() - t0).count();
            perf.disable();
            hw_cache  += perf.cache_misses();
            hw_instr  += perf.instructions();
            hw_cycles += perf.cycles();
            hw_brmiss += perf.branch_misses();
            in_retrain_window = false;
            retrain_times_ms.push_back(ms);
            retrain_ms_total += ms;
            ++retrain_count;
        }
    }

    bool lookup(uint64_t key) {
        bool sample = ((op_count++ % sample_rate) == 0);
        auto qt0 = sample ? Clock::now() : Clock::time_point{};

        bool found;
        if (index.segments.empty()) {
            found = std::binary_search(data.begin(), data.end(), key);
        } else {
            auto r = index.search_range(key);
            r.lo = std::max(int64_t(0), std::min(r.lo, static_cast<int64_t>(data.size())));
            r.hi = std::max(r.lo, std::min(r.hi, static_cast<int64_t>(data.size())));
            auto* p = std::lower_bound(
                data.data() + r.lo, data.data() + r.hi, key);
            found = (p != data.data() + r.hi && *p == key);
        }

        if (sample) {
            double lat = Ns(Clock::now() - qt0).count();
            lookup_lat_ns.push_back(lat);
            if (in_retrain_window)
                lookup_lat_retrain_ns.push_back(lat);
        }

        return found;
    }
};

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int64_t     epsilon      = std::stoll(get_arg(argc, argv, "--epsilon", "64"));
    std::string algo_s       = get_arg(argc, argv, "--algo",          "optimal");
    int         threads      = std::stoi(get_arg(argc, argv, "--threads",     "1"));
    size_t      n            = std::stoull(get_arg(argc, argv, "--n",    "1000000"));
    double      insert_ratio = std::stod(get_arg(argc, argv, "--insert-ratio","0.5"));
    std::string workload     = get_arg(argc, argv, "--workload",      "balanced");
    size_t      sample_rate  = std::stoull(get_arg(argc, argv, "--sample-rate","100"));
    std::string exp_id       = get_arg(argc, argv, "--exp-id",        "dynamic");
    std::string dataset      = get_arg(argc, argv, "--dataset",       "");
    if (threads > 1)
        std::cerr << "[WARN] dynamic_bench: --threads=" << threads
                  << " ignored; NaiveDynamic is single-threaded\n";
    (void)threads;

    // Workload drives insert_ratio when explicitly set.
    if (workload == "readonly")    insert_ratio = 0.0;
    else if (workload == "write_heavy") insert_ratio = 0.9;
    else if (workload == "balanced")    insert_ratio = 0.5;
    // If --insert-ratio was passed explicitly, it overrides workload name above
    // only when workload is still the default "balanced".

    pla::PlaAlgo    algo = pla::algo_from_string(algo_s);
    pla::PlaOptions opts;
    opts.threads = static_cast<unsigned>(1); // NaiveDynamic is single-threaded

    // ── Generate or load initial dataset (50 % of n_keys) ────────────────────
    std::mt19937_64 rng(42);
    std::vector<uint64_t> insert_stream;
    std::string ds_label;

    size_t n_initial = n / 2;
    std::vector<uint64_t> init_keys;

    if (!dataset.empty()) {
        // Read at most n entries from disk — avoids loading 200M keys when --n 5M.
        std::cerr << "[dynamic] Loading dataset (max " << n << " keys): " << dataset << "\n" << std::flush;
        auto all_keys = load_binary(dataset, n);
        std::cerr << "[dynamic] Sorting " << all_keys.size() << " keys...\n" << std::flush;
        std::sort(all_keys.begin(), all_keys.end());
        all_keys.erase(std::unique(all_keys.begin(), all_keys.end()), all_keys.end());
        size_t n_total = all_keys.size();  // actual after dedup (≤ n)
        n_initial = std::min(n / 2, n_total);
        init_keys.assign(all_keys.begin(), all_keys.begin() + static_cast<long>(n_initial));
        insert_stream.assign(all_keys.begin() + static_cast<long>(n_initial), all_keys.end());
        // Do NOT override n with n_total here — that was causing 200M ops when --n 5M was passed.
        // n stays as the user-requested cap; n_total is just the dedup-reduced count.
        n = n_total;
        ds_label = dataset;
        std::cerr << "[dynamic] n=" << n << " init=" << n_initial
                  << " insert_stream=" << insert_stream.size() << "\n" << std::flush;
    } else {
        init_keys.resize(n_initial);
        for (auto& k : init_keys) k = rng();
        std::sort(init_keys.begin(), init_keys.end());
        init_keys.erase(std::unique(init_keys.begin(), init_keys.end()), init_keys.end());
        ds_label = "synth_uniform_" + std::to_string(n);
    }
    n_initial = init_keys.size();

    size_t rss_before = get_rss_mb();

    NaiveDynamic dyn(epsilon, algo, opts, sample_rate);

    // ── Bulk-load ─────────────────────────────────────────────────────────────
    auto t_build0 = Clock::now();
    dyn.bulk_load(init_keys);
    double build_ms = Ms(Clock::now() - t_build0).count();

    // ── Mixed workload ────────────────────────────────────────────────────────
    size_t n_ops    = n;
    size_t n_insert = static_cast<size_t>(n_ops * insert_ratio);
    size_t n_lookup = n_ops - n_insert;

    volatile int64_t sink = 0;
    auto t0 = Clock::now();

    size_t ins_done = 0, look_done = 0;
    double total_lookup_ns = 0.0;
    size_t report_interval = std::max(n_ops / 10, size_t(50000));

    for (size_t op = 0; op < n_ops; ++op) {
        if (op > 0 && op % report_interval == 0) {
            double pct    = 100.0 * op / n_ops;
            double elapsed = Ms(Clock::now() - t0).count() / 1000.0;
            std::cerr << "[dynamic] " << std::fixed << std::setprecision(1)
                      << pct << "% op=" << op << "/" << n_ops
                      << " ins=" << ins_done << " elapsed=" << elapsed << "s\n" << std::flush;
        }
        uint64_t key;
        if (!insert_stream.empty() && ins_done < insert_stream.size())
            key = insert_stream[ins_done % insert_stream.size()];
        else
            key = rng();

        bool do_insert = (ins_done < n_insert) &&
                         (look_done >= n_lookup || (op % 2 == 0));
        if (do_insert) {
            dyn.insert(key);
            ++ins_done;
        } else if (look_done < n_lookup) {
            auto lt0 = Clock::now();
            sink ^= dyn.lookup(key) ? 1 : 0;
            total_lookup_ns += Ns(Clock::now() - lt0).count();
            ++look_done;
        }
    }

    size_t rss_after = get_rss_mb();
    int64_t rss_mb = static_cast<int64_t>(rss_after) - static_cast<int64_t>(rss_before);

    double elapsed_ms = Ms(Clock::now() - t0).count();
    // ops_s: combined throughput over total wall-clock (includes inserts + retrains).
    double ops_s = n_ops / (elapsed_ms / 1000.0);
    // lookup_ops_s: lookup-only throughput, excluding insert and retrain cost.
    double lookup_ops_s = (look_done > 0 && total_lookup_ns > 0)
                        ? (look_done / (total_lookup_ns / 1e9))
                        : 0.0;
    (void)sink;

    // ── Latency percentiles ────────────────────────────────────────────────────
    double p50 = vec_pct(dyn.lookup_lat_ns, 50.0);
    double p95 = vec_pct(dyn.lookup_lat_ns, 95.0);
    double p99 = vec_pct(dyn.lookup_lat_ns, 99.0);

    // Retrain-window latency p99 (0 if no retrains or no samples during retrain).
    double retrain_window_p99 = vec_pct(dyn.lookup_lat_retrain_ns, 99.0);

    // Per-retrain time statistics.
    double retrain_p50 = vec_pct(dyn.retrain_times_ms, 50.0);
    double retrain_p95 = vec_pct(dyn.retrain_times_ms, 95.0);

    // Workload label for JSONL.
    std::string wl_label = workload + "_ir" +
        std::to_string(static_cast<int>(insert_ratio * 100));

    std::cout << std::fixed << std::setprecision(3)
        << "{"
        << "\"exp_id\":\""            << exp_id         << "\","
        << "\"scenario\":\"dynamic\","
        << "\"index\":\"LOFT-naive\","
        << "\"pla\":\""               << algo_s         << "\","
        << "\"epsilon\":"             << epsilon        << ","
        << "\"threads\":"             << 1              << ","
        << "\"dataset\":\""          << ds_label      << "\","
        << "\"workload\":\""          << wl_label       << "\","
        << "\"build_ms\":"            << build_ms       << ","
        << "\"retrain_ms\":"          << dyn.retrain_ms_total << ","
        << "\"retrain_count\":"       << dyn.retrain_count    << ","
        << "\"retrain_p50_ms\":"      << retrain_p50    << ","
        << "\"retrain_p95_ms\":"      << retrain_p95    << ","
        << "\"retrain_window_p99_ns\":"<< retrain_window_p99 << ","
        << "\"seg_cnt\":"             << dyn.index.segments.size() << ","
        << "\"bytes_index\":"         << dyn.index.bytes()         << ","
        << "\"n_insert\":"            << ins_done       << ","
        << "\"n_lookup\":"            << look_done      << ","
        // ops_s: all ops / total wall-clock (includes insert O(n) and retrain cost).
        << "\"ops_s\":"               << ops_s          << ","
        // lookup_ops_s: pure lookup throughput, uncontaminated by insert/retrain time.
        << "\"lookup_ops_s\":"        << lookup_ops_s   << ","
        << "\"p50_ns\":"              << p50            << ","
        << "\"p95_ns\":"              << p95            << ","
        << "\"p99_ns\":"              << p99            << ","
        << "\"cache_misses\":"   << dyn.hw_cache << ","
        << "\"branches\":0,"
        << "\"branch_misses\":"  << dyn.hw_brmiss << ","
        << "\"instructions\":"   << dyn.hw_instr  << ","
        << "\"cycles\":"         << dyn.hw_cycles << ","
        << "\"rss_mb\":"        << rss_mb        << ","
        << "\"fetch_strategy\":-1,\"io_pages\":0"
        << "}\n";
    return 0;
}
