// bench/inmem/lookup_bench.cpp
// End-to-end in-memory lookup benchmark using PLA index.
// Supports: random, sequential, Zipf query distributions.
// Output: JSONL with latency p50/p95/p99 and throughput.

#include <pla/pla_api.h>

#include <algorithm>
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

using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::duration<double, std::nano>;

// ─── CLI ─────────────────────────────────────────────────────────────────────
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

// ─── Zipf query generator ────────────────────────────────────────────────────
struct ZipfGen {
    std::vector<size_t> table;
    std::mt19937_64 rng;
    ZipfGen(size_t n, double alpha, uint64_t seed = 42) : rng(seed) {
        // Alias method approximation via rejection.
        table.resize(n);
        std::vector<double> w(n);
        double sum = 0;
        for (size_t i = 0; i < n; ++i) { w[i] = 1.0 / std::pow(i+1, alpha); sum += w[i]; }
        for (size_t i = 0; i < n; ++i) w[i] /= sum;
        // Store CDF-based table for O(log n) sampling.
        // (Full alias method overkill; use binary search on CDF.)
        std::vector<double> cdf(n);
        std::partial_sum(w.begin(), w.end(), cdf.begin());
        for (size_t i = 0; i < n; ++i) {
            std::uniform_real_distribution<double> u(0,1);
            double r = u(rng);
            size_t idx = (size_t)(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
            table[i] = std::min(idx, n-1);
        }
    }
    size_t operator()(size_t /*query_idx*/) {
        static thread_local size_t pos = 0;
        return table[pos++ % table.size()];
    }
};

// ─── Binary search within range ──────────────────────────────────────────────
static int64_t range_search(const uint64_t* keys, int64_t lo, int64_t hi, uint64_t target) {
    // Inclusive [lo, hi).
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (keys[mid] < target)      lo = mid + 1;
        else if (keys[mid] > target) hi = mid;
        else                         return mid;
    }
    return -1; // not found
}

// ─── Percentile helper ───────────────────────────────────────────────────────
static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[std::min(idx, v.size()-1)];
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int64_t    epsilon   = std::stoll(get_arg(argc, argv, "--epsilon", "64"));
    std::string algo_s   = get_arg(argc, argv, "--algo", "optimal");
    int threads          = std::stoi(get_arg(argc, argv, "--threads", "1"));
    std::string dataset  = get_arg(argc, argv, "--dataset", "");
    size_t n_synth       = std::stoull(get_arg(argc, argv, "--n", "1000000"));
    std::string workload = get_arg(argc, argv, "--workload", "readonly");
    size_t n_queries     = std::stoull(get_arg(argc, argv, "--queries", "1000000"));
    double zipf_alpha    = std::stod(get_arg(argc, argv, "--zipf-alpha", "1.0"));
    std::string exp_id   = get_arg(argc, argv, "--exp-id", "inmem");
    bool verbose         = has_flag(argc, argv, "--verbose");

    // Load / generate keys.
    std::vector<uint64_t> keys;
    std::string ds_name;
    if (!dataset.empty()) {
        keys = load_binary(dataset);
        ds_name = dataset;
    } else {
        keys = gen_uniform_sorted(n_synth);
        ds_name = "synth_uniform_" + std::to_string(n_synth);
    }
    std::sort(keys.begin(), keys.end());
    const size_t n = keys.size();

    // Build PLA.
    pla::PlaAlgo algo = pla::algo_from_string(algo_s);
    pla::PlaOptions opts;
    opts.threads = static_cast<unsigned>(threads);

    auto t_build0 = Clock::now();
    pla::PlaResult index = pla::build_pla(keys, epsilon, algo, opts);
    double build_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_build0).count();

    if (verbose)
        std::cerr << "Built " << index.segments.size() << " segments in " << build_ms << " ms\n";

    // Build query set.
    // workload: readonly (lookup existing), balanced (50% lookup + 50% miss), zipf
    std::vector<uint64_t> queries;
    queries.reserve(n_queries);
    std::mt19937_64 rng(123);

    if (workload == "zipf") {
        // Generate pre-computed Zipf indices.
        ZipfGen zgen(n, zipf_alpha);
        for (size_t q = 0; q < n_queries; ++q)
            queries.push_back(keys[zgen(q)]);
    } else if (workload == "balanced") {
        std::uniform_int_distribution<size_t> idx_dist(0, n-1);
        std::uniform_int_distribution<uint64_t> miss_dist;
        for (size_t q = 0; q < n_queries; ++q) {
            if (q % 2 == 0) queries.push_back(keys[idx_dist(rng)]);
            else             queries.push_back(miss_dist(rng)); // likely miss
        }
    } else { // readonly
        std::uniform_int_distribution<size_t> idx_dist(0, n-1);
        for (size_t q = 0; q < n_queries; ++q)
            queries.push_back(keys[idx_dist(rng)]);
    }

    // Warm-up (10% of queries).
    volatile int64_t sink = 0;
    for (size_t q = 0; q < n_queries / 10; ++q) {
        auto r = index.search_range(queries[q]);
        sink ^= range_search(keys.data(), r.lo, r.hi, queries[q]);
    }
    (void)sink;

    // Timed run.
    std::vector<double> latencies;
    latencies.reserve(n_queries);

    auto t0 = Clock::now();
    for (size_t q = 0; q < n_queries; ++q) {
        auto qt0 = Clock::now();
        auto r   = index.search_range(queries[q]);
        sink ^= range_search(keys.data(), r.lo, r.hi, queries[q]);
        latencies.push_back(std::chrono::duration<double, std::nano>(Clock::now() - qt0).count());
    }
    double total_ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
    (void)sink;

    double p50 = percentile(latencies, 50.0);
    double p95 = percentile(latencies, 95.0);
    double p99 = percentile(latencies, 99.0);
    double ops_s = (n_queries / total_ns) * 1e9;

    std::cout << std::fixed << std::setprecision(3)
        << "{"
        << "\"exp_id\":\""    << exp_id << "\","
        << "\"scenario\":\"inmem\","
        << "\"index\":\"PGM-index\","
        << "\"pla\":\""       << algo_s   << "\","
        << "\"epsilon\":"     << epsilon  << ","
        << "\"threads\":"     << threads  << ","
        << "\"dataset\":\""   << ds_name  << "\","
        << "\"workload\":\""  << workload << "\","
        << "\"build_ms\":"    << build_ms << ","
        << "\"seg_cnt\":"     << index.segments.size() << ","
        << "\"bytes_index\":" << index.bytes()         << ","
        << "\"ops_s\":"       << ops_s    << ","
        << "\"p50_ns\":"      << p50      << ","
        << "\"p95_ns\":"      << p95      << ","
        << "\"p99_ns\":"      << p99      << ","
        << "\"cache_misses\":0,\"branches\":0,\"branch_misses\":0,"
        << "\"instructions\":0,\"cycles\":0,\"rss_mb\":0,"
        << "\"fetch_strategy\":-1,\"io_pages\":0"
        << "}\n";
    return 0;
}
