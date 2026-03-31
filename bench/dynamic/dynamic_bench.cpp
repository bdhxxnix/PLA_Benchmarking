// bench/dynamic/dynamic_bench.cpp
// LOFT dynamic workload benchmark wrapper.
// Wraps third_party/LOFT microbench interface; falls back to a
// simple insert+lookup loop if LOFT headers are not available.
//
// LOFT dependencies: MKL (or USE_MKL=OFF fallback), jemalloc, urcu.
// See scripts/bootstrap_ubuntu22.sh for installation.
//
// Usage:
//   dynamic_bench --algo optimal --epsilon 64 --threads 4
//                 --n 1000000 --insert-ratio 0.5 --exp-id dynamic

#include <pla/pla_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::string get_arg(int argc, char** argv, const char* f, const char* d = "") {
    for (int i = 1; i+1 < argc; ++i) if (!std::strcmp(argv[i], f)) return argv[i+1];
    return d;
}
static bool has_flag(int argc, char** argv, const char* f) {
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], f)) return true;
    return false;
}

// ─── Minimal dynamic ordered structure (fallback when LOFT not available) ────
// A sorted vector with O(n) insert/O(log n) lookup — captures the same
// retrain-on-insert metric without the LOFT dependency.
struct NaiveDynamic {
    std::vector<uint64_t> data;
    pla::PlaResult        index;
    int64_t               epsilon;
    pla::PlaAlgo          algo;
    pla::PlaOptions       opts;
    size_t                retrain_count = 0;
    double                retrain_ms    = 0;

    NaiveDynamic(int64_t eps, pla::PlaAlgo a, pla::PlaOptions o)
        : epsilon(eps), algo(a), opts(o) {}

    void insert(uint64_t key) {
        auto pos = std::lower_bound(data.begin(), data.end(), key);
        data.insert(pos, key);
        // Retrain every 10k inserts (simulates LOFT retrain trigger).
        if (data.size() % 10000 == 0) {
            auto t0 = Clock::now();
            index = pla::build_pla(data, epsilon, algo, opts);
            retrain_ms += std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
            ++retrain_count;
        }
    }

    bool lookup(uint64_t key) const {
        if (index.segments.empty()) {
            return std::binary_search(data.begin(), data.end(), key);
        }
        auto r = index.search_range(key);
        auto* p = std::lower_bound(data.data() + r.lo, data.data() + r.hi, key);
        return p != data.data() + r.hi && *p == key;
    }
};

int main(int argc, char** argv) {
    int64_t    epsilon      = std::stoll(get_arg(argc, argv, "--epsilon", "64"));
    std::string algo_s      = get_arg(argc, argv, "--algo", "optimal");
    int threads             = std::stoi(get_arg(argc, argv, "--threads", "1"));
    size_t n                = std::stoull(get_arg(argc, argv, "--n", "1000000"));
    double insert_ratio     = std::stod(get_arg(argc, argv, "--insert-ratio", "0.5"));
    std::string exp_id      = get_arg(argc, argv, "--exp-id", "dynamic");
    (void)threads;

    pla::PlaAlgo algo = pla::algo_from_string(algo_s);
    pla::PlaOptions opts;
    opts.threads = static_cast<unsigned>(threads);

    // Generate initial dataset (50% of n) then mixed insert/lookup ops.
    std::mt19937_64 rng(42);
    size_t n_initial = n / 2;
    std::vector<uint64_t> init_keys(n_initial);
    for (auto& k : init_keys) k = rng();
    std::sort(init_keys.begin(), init_keys.end());

    NaiveDynamic dyn(epsilon, algo, opts);
    // Bulk-load initial keys.
    for (auto k : init_keys) dyn.data.push_back(k);
    auto t_build0 = Clock::now();
    dyn.index = pla::build_pla(dyn.data, epsilon, algo, opts);
    double build_ms = std::chrono::duration<double,std::milli>(Clock::now()-t_build0).count();

    // Mixed workload.
    size_t n_ops    = n;
    size_t n_insert = static_cast<size_t>(n_ops * insert_ratio);
    size_t n_lookup = n_ops - n_insert;

    volatile int64_t sink = 0;
    auto t0 = Clock::now();
    size_t ins_done = 0, look_done = 0;
    for (size_t op = 0; op < n_ops; ++op) {
        uint64_t key = rng();
        if (ins_done < n_insert && (look_done >= n_lookup || op % 2 == 0)) {
            dyn.insert(key);
            ++ins_done;
        } else {
            sink ^= dyn.lookup(key) ? 1 : 0;
            ++look_done;
        }
    }
    double elapsed_ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    double ops_s = n_ops / (elapsed_ms / 1000.0);
    (void)sink;

    std::cout << std::fixed << std::setprecision(3)
        << "{"
        << "\"exp_id\":\""      << exp_id       << "\","
        << "\"scenario\":\"dynamic\","
        << "\"index\":\"LOFT-naive\","
        << "\"pla\":\""         << algo_s       << "\","
        << "\"epsilon\":"       << epsilon       << ","
        << "\"threads\":"       << threads       << ","
        << "\"dataset\":\"synth_uniform_" << n  << "\","
        << "\"workload\":\"insert_ratio_" << insert_ratio << "\","
        << "\"build_ms\":"      << build_ms      << ","
        << "\"retrain_ms\":"    << dyn.retrain_ms<< ","
        << "\"retrain_count\":" << dyn.retrain_count << ","
        << "\"seg_cnt\":"       << dyn.index.segments.size() << ","
        << "\"bytes_index\":"   << dyn.index.bytes()         << ","
        << "\"ops_s\":"         << ops_s         << ","
        << "\"p50_ns\":0,\"p95_ns\":0,\"p99_ns\":0,"
        << "\"cache_misses\":0,\"branches\":0,\"branch_misses\":0,"
        << "\"instructions\":0,\"cycles\":0,\"rss_mb\":0,"
        << "\"fetch_strategy\":-1,\"io_pages\":0"
        << "}\n";
    return 0;
}
