// bench/dynamic/dynamic_bench.cpp
// LOFT dynamic workload benchmark (DW-A and DW-B experiments).
//
// Uses the real LOFT engine from third_party/LOFT when URCU is available.
// Falls back to NaiveDynamic (sorted vector + periodic retrain) otherwise.
//
// Workload modes (--workload):
//   readonly    — 0 % inserts  (pure lookup after bulk-load)
//   write_heavy — 90 % inserts (DW-A: model cost under heavy churn)
//   balanced    — 50 % inserts (DW-B default)
//
// LOFT background threads (--bg-n):
//   0 (default) — no background SMO, single-threaded, no retrain
//   >0          — bg threads handle split/merge/expand (retrain_cost measured)

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
#include <thread>
#include <vector>

// LOFT (real third-party) — requires URCU
// LOFT_impl.h must be included BEFORE LOFT.h (template definitions)
#if __has_include(<LOFT_impl.h>) && __has_include(<LOFT.h>)
#  include <LOFT_impl.h>
#  include <LOFT.h>
#  define HAVE_LOFT 1
#else
#  define HAVE_LOFT 0
#endif

using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::duration<double, std::nano>;
using Ms    = std::chrono::duration<double, std::milli>;

#include "perf_counters.h"
#include "rss.h"

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

// ══════════════════════════════════════════════════════════════════════════════
// Real LOFT wrapper (third-party)
// ══════════════════════════════════════════════════════════════════════════════

#if HAVE_LOFT

struct LoftWrapper {
    using LoftIndex = loft::LOFT<uint64_t, uint64_t>;

    LoftIndex*       idx  = nullptr;
    size_t           n_initial = 0;
    int64_t          epsilon   = 64;
    size_t           bg_n      = 0;
    uint8_t          worker_id = 0;

    // Build time (constructor duration).
    double build_ms = 0;

    // Retrain statistics (populated only when bg_n > 0).
    size_t retrain_count     = 0;
    double retrain_ms_total  = 0;
    std::vector<double> retrain_times_ms;

    // Latency samples.
    std::vector<double> lookup_lat_ns;
    std::vector<double> lookup_lat_retrain_ns;
    size_t sample_rate = 100;
    size_t op_count    = 0;
    bool   in_retrain  = false;

    // Accumulated hardware counters.
    PerfCounters perf;
    int64_t hw_cache  = 0;
    int64_t hw_instr  = 0;
    int64_t hw_cycles = 0;
    int64_t hw_brmiss = 0;

    LoftWrapper(int64_t eps, size_t bg, size_t sr = 100)
        : epsilon(eps), bg_n(bg), sample_rate(sr) {
        perf.open_all();
    }

    ~LoftWrapper() { delete idx; }

    void bulk_load(const std::vector<uint64_t>& init_keys) {
        n_initial = init_keys.size();

        // LOFT requires strictly increasing keys, unique values
        std::vector<uint64_t> vals(init_keys.size());
        for (size_t i = 0; i < init_keys.size(); ++i)
            vals[i] = i;  // value = position

        rcu_register_thread();

        perf.reset();
        perf.enable();
        auto t0 = Clock::now();

        // work_num = 1 (single foreground thread), bg_n controls background
        idx = new LoftIndex(init_keys, vals, /*work_num=*/1, bg_n);

        build_ms = Ms(Clock::now() - t0).count();
        perf.disable();

        hw_cache  += perf.cache_misses();
        hw_instr  += perf.instructions();
        hw_cycles += perf.cycles();
        hw_brmiss += perf.branch_misses();

        // The constructor counts as the initial build (retrain #0)
        retrain_times_ms.push_back(build_ms);
        retrain_ms_total += build_ms;
        ++retrain_count;
    }

    void insert(uint64_t key) {
        uint64_t val = 0;  // dummy value
        auto t0 = Clock::now();
        idx->insert(key, val, worker_id);
        // Note: LOFT insert timing includes possible CAS retry overhead
    }

    bool lookup(uint64_t key) {
        bool sample = ((op_count++ % sample_rate) == 0);
        auto qt0 = sample ? Clock::now() : Clock::time_point{};

        uint64_t val = 0;
        bool found = idx->query(key, val, worker_id);

        if (sample) {
            double lat = Ns(Clock::now() - qt0).count();
            lookup_lat_ns.push_back(lat);
            if (in_retrain)
                lookup_lat_retrain_ns.push_back(lat);
        }
        return found;
    }

    size_t seg_cnt() const {
        // LOFT doesn't expose segment count per se; report 0
        return 0;
    }

    size_t bytes() const {
        // LOFT doesn't expose size_in_bytes easily
        return 0;
    }
};

#endif // HAVE_LOFT


// ══════════════════════════════════════════════════════════════════════════════
// NaiveDynamic fallback (when LOFT / URCU not available)
// ══════════════════════════════════════════════════════════════════════════════

struct NaiveDynamic {
    std::vector<uint64_t> data;
    pla::PlaResult        index;
    int64_t               epsilon;
    pla::PlaAlgo          algo;
    pla::PlaOptions       opts;

    size_t retrain_count     = 0;
    double retrain_ms_total  = 0;
    std::vector<double> retrain_times_ms;

    std::vector<double> lookup_lat_ns;
    std::vector<double> lookup_lat_retrain_ns;
    size_t sample_rate = 100;
    size_t op_count    = 0;

    bool in_retrain_window = false;

    static constexpr size_t RETRAIN_INTERVAL = 10000;

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


// ══════════════════════════════════════════════════════════════════════════════
// Unified dispatch (template on index type)
// ══════════════════════════════════════════════════════════════════════════════

template<typename Index>
void run_workload(Index& dyn,
                  const std::vector<uint64_t>& init_keys,
                  size_t n_ops, size_t n_insert, size_t n_lookup,
                  const std::vector<uint64_t>& insert_stream,
                  std::mt19937_64& rng,
                  volatile int64_t& sink,
                  size_t& ins_done, size_t& look_done) {
    size_t report_interval = std::max(n_ops / 10, size_t(50000));

    for (size_t op = 0; op < n_ops; ++op) {
        if (op > 0 && op % report_interval == 0) {
            std::cerr << "[dynamic] " << std::fixed << std::setprecision(0)
                      << (100.0 * op / n_ops) << "% op=" << op << "/" << n_ops
                      << " ins=" << ins_done << "\n" << std::flush;
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
            sink ^= dyn.lookup(key) ? 1 : 0;
            ++look_done;
        }
    }
}


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
    size_t      bg_n         = std::stoull(get_arg(argc, argv, "--bg-n",          "0"));
    std::string index_type   = get_arg(argc, argv, "--index",         "loft");

    if (threads > 1)
        std::cerr << "[WARN] dynamic_bench: --threads=" << threads
                  << " ignored; single-threaded workload\n";
    (void)threads;

    // Workloads
    if (workload == "readonly")           insert_ratio = 0.0;
    else if (workload == "write_heavy")   insert_ratio = 0.9;
    else if (workload == "balanced")      insert_ratio = 0.5;

    pla::PlaAlgo    algo = pla::algo_from_string(algo_s);
    pla::PlaOptions opts; opts.threads = 1;

    // ── Generate or load initial dataset ────────────────────────────────────
    std::mt19937_64 rng(42);
    std::vector<uint64_t> insert_stream;
    std::string ds_label;

    size_t n_initial = n / 2;
    std::vector<uint64_t> init_keys;

    if (!dataset.empty()) {
        std::cerr << "[dynamic] Loading dataset (max " << n << " keys): "
                  << dataset << "\n" << std::flush;
        auto all_keys = load_binary(dataset, n);
        std::cerr << "[dynamic] Sorting " << all_keys.size()
                  << " keys...\n" << std::flush;
        std::sort(all_keys.begin(), all_keys.end());
        all_keys.erase(std::unique(all_keys.begin(), all_keys.end()),
                       all_keys.end());
        size_t n_total = all_keys.size();
        n_initial = std::min(n / 2, n_total);
        init_keys.assign(all_keys.begin(),
                         all_keys.begin() + static_cast<long>(n_initial));
        insert_stream.assign(
            all_keys.begin() + static_cast<long>(n_initial), all_keys.end());
        n = n_total;
        ds_label = dataset;
        std::cerr << "[dynamic] n=" << n << " init=" << n_initial
                  << " insert_stream=" << insert_stream.size() << "\n"
                  << std::flush;
    } else {
        init_keys.resize(n_initial);
        for (auto& k : init_keys) k = rng();
        std::sort(init_keys.begin(), init_keys.end());
        init_keys.erase(std::unique(init_keys.begin(), init_keys.end()),
                        init_keys.end());
        ds_label = "synth_uniform_" + std::to_string(n);
    }
    n_initial = init_keys.size();

    size_t rss_before = get_rss_mb();

    // ── Build index ─────────────────────────────────────────────────────────
    // Common metric variables (populated by both code paths)
    double build_ms    = 0;
    double ops_s       = 0;
    double lookup_ops_s = 0;
    double p50 = 0, p95 = 0, p99 = 0;
    double retrain_p50 = 0, retrain_p95 = 0;
    double retrain_window_p99 = 0;
    size_t retrain_count     = 0;
    double retrain_ms_total  = 0;
    size_t n_insert_done = 0, n_lookup_done = 0;
    int64_t hw_cache = 0, hw_instr = 0, hw_cycles = 0, hw_brmiss = 0;
    std::vector<double> retrain_times;
    std::vector<double> lookup_lats;
    std::string index_name;

    volatile int64_t sink = 0;

    size_t n_ops    = n;
    size_t n_insert = static_cast<size_t>(n_ops * insert_ratio);
    size_t n_lookup = n_ops - n_insert;

#if HAVE_LOFT
    if (index_type == "loft") {
        LoftWrapper dyn(epsilon, bg_n, sample_rate);
        index_name = "LOFT";

        // Bulk-load
        auto t0 = Clock::now();
        dyn.bulk_load(init_keys);
        build_ms = Ms(Clock::now() - t0).count();

        // Workload
        auto tw0 = Clock::now();
        run_workload(dyn, init_keys, n_ops, n_insert, n_lookup,
                     insert_stream, rng, sink, n_insert_done, n_lookup_done);
        double elapsed_ms = Ms(Clock::now() - tw0).count();

        ops_s        = n_ops / (elapsed_ms / 1000.0);
        lookup_ops_s = n_lookup_done / (elapsed_ms / 1000.0);
        retrain_count    = dyn.retrain_count;
        retrain_ms_total = dyn.retrain_ms_total;
        retrain_times    = dyn.retrain_times_ms;
        lookup_lats      = dyn.lookup_lat_ns;
        hw_cache  = dyn.hw_cache;
        hw_instr  = dyn.hw_instr;
        hw_cycles = dyn.hw_cycles;
        hw_brmiss = dyn.hw_brmiss;
    } else
#endif
    {
        NaiveDynamic dyn(epsilon, algo, opts, sample_rate);
        index_name = "NaiveDynamic";

        auto t0 = Clock::now();
        dyn.bulk_load(init_keys);
        build_ms = Ms(Clock::now() - t0).count();

        auto tw0 = Clock::now();
        run_workload(dyn, init_keys, n_ops, n_insert, n_lookup,
                     insert_stream, rng, sink, n_insert_done, n_lookup_done);
        double elapsed_ms = Ms(Clock::now() - tw0).count();

        ops_s        = n_ops / (elapsed_ms / 1000.0);
        lookup_ops_s = n_lookup_done / (elapsed_ms / 1000.0);
        retrain_count    = dyn.retrain_count;
        retrain_ms_total = dyn.retrain_ms_total;
        retrain_times    = dyn.retrain_times_ms;
        lookup_lats      = dyn.lookup_lat_ns;
        hw_cache  = dyn.hw_cache;
        hw_instr  = dyn.hw_instr;
        hw_cycles = dyn.hw_cycles;
        hw_brmiss = dyn.hw_brmiss;
    }

    size_t rss_after = get_rss_mb();
    int64_t rss_mb = static_cast<int64_t>(rss_after) - static_cast<int64_t>(rss_before);

    // ── Compute percentiles ─────────────────────────────────────────────────
    if (!lookup_lats.empty()) {
        p50 = vec_pct(lookup_lats, 50.0);
        p95 = vec_pct(lookup_lats, 95.0);
        p99 = vec_pct(lookup_lats, 99.0);
    }
    if (!retrain_times.empty()) {
        retrain_p50 = vec_pct(retrain_times, 50.0);
        retrain_p95 = vec_pct(retrain_times, 95.0);
    }

    // ── JSONL output ────────────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(3)
        << "{"
        << "\"exp_id\":\""         << exp_id          << "\","
        << "\"scenario\":\"dynamic\","
        << "\"index\":\""          << index_name      << "\","
        << "\"pla\":\""            << algo_s          << "\","
        << "\"epsilon\":"          << epsilon         << ","
        << "\"threads\":"          << 1               << ","
        << "\"dataset\":\""        << ds_label        << "\","
        << "\"workload\":\""       << workload        << "\","
        << "\"build_ms\":"         << build_ms        << ","
        << "\"seg_cnt\":"          << 0               << ","
        << "\"bytes_index\":"      << 0               << ","
        << "\"ops_s\":"            << ops_s           << ","
        << "\"p50_ns\":"           << p50             << ","
        << "\"p95_ns\":"           << p95             << ","
        << "\"p99_ns\":"           << p99             << ","
        << "\"n_insert\":"         << n_insert_done   << ","
        << "\"n_lookup\":"         << n_lookup_done   << ","
        << "\"lookup_ops_s\":"     << lookup_ops_s    << ","
        << "\"retrain_ms\":"       << retrain_ms_total << ","
        << "\"retrain_count\":"    << retrain_count   << ","
        << "\"retrain_p50_ms\":"   << retrain_p50     << ","
        << "\"retrain_p95_ms\":"   << retrain_p95     << ","
        << "\"retrain_window_p99_ns\":0,"
        << "\"cache_misses\":"    << hw_cache        << ","
        << "\"branches\":0,"
        << "\"branch_misses\":"   << hw_brmiss       << ","
        << "\"instructions\":"    << hw_instr        << ","
        << "\"cycles\":"          << hw_cycles       << ","
        << "\"rss_mb\":"          << rss_mb          << ","
        << "\"fetch_strategy\":-1,\"io_pages\":0"
        << "}\n";

    (void)sink;
    return 0;
}
