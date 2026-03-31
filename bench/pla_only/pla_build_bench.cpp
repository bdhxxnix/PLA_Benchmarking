// bench/pla_only/pla_build_bench.cpp
// PLA-only construction benchmark.
// Measures: build_ms, seg_cnt, bytes_index, max_err (verified <= epsilon).
// Output: one JSONL line to stdout (captured by runner).

#include <pla/pla_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

// ─── Dataset loaders ─────────────────────────────────────────────────────────
static std::vector<uint64_t> load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }
    auto sz = f.tellg(); f.seekg(0);
    size_t n = sz / sizeof(uint64_t);
    std::vector<uint64_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), sz);
    return v;
}

static std::vector<uint64_t> gen_uniform(size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = rng();
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

static std::vector<uint64_t> gen_lognormal(size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::lognormal_distribution<double> dist(0.0, 2.0);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = static_cast<uint64_t>(dist(rng) * 1e9);
    std::sort(v.begin(), v.end());
    return v;
}

// ─── Timing helpers ──────────────────────────────────────────────────────────
using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // --epsilon <int>     (default 64)
    // --algo    optimal|swing|greedy
    // --threads <int>     (default 1)
    // --dataset <path>    binary uint64 file; if missing generates uniform 1M
    // --n       <int>     number of keys if synthetic (default 1000000)
    // --dist    uniform|lognormal
    // --dup-handling      enable duplicate handling (default on)
    // --no-verify         skip epsilon verification
    // --exp-id  <str>     experiment id for JSONL

    int64_t    epsilon = std::stoll(get_arg(argc, argv, "--epsilon", "64"));
    std::string algo_s = get_arg(argc, argv, "--algo", "optimal");
    int threads        = std::stoi(get_arg(argc, argv, "--threads", "1"));
    std::string dataset= get_arg(argc, argv, "--dataset", "");
    size_t n_synth     = std::stoull(get_arg(argc, argv, "--n", "1000000"));
    std::string dist_s = get_arg(argc, argv, "--dist", "uniform");
    bool no_verify     = has_flag(argc, argv, "--no-verify");
    std::string exp_id = get_arg(argc, argv, "--exp-id", "pla_only");

    // Load or generate keys.
    std::vector<uint64_t> keys;
    if (!dataset.empty()) {
        keys = load_binary(dataset);
        std::sort(keys.begin(), keys.end());
    } else if (dist_s == "lognormal") {
        keys = gen_lognormal(n_synth);
    } else {
        keys = gen_uniform(n_synth);
    }

    pla::PlaAlgo algo = pla::algo_from_string(algo_s);
    pla::PlaOptions opts;
    opts.threads = static_cast<unsigned>(threads);
    opts.handle_duplicates = !has_flag(argc, argv, "--no-dup-handling");

    // Run build (repeat 3x, report median).
    std::vector<double> times;
    pla::PlaResult result;
    for (int rep = 0; rep < 3; ++rep) {
        result = pla::build_pla(keys, epsilon, algo, opts);
        times.push_back(result.build_ms);
    }
    std::sort(times.begin(), times.end());
    double build_ms = times[times.size()/2];

    // Verify epsilon guarantee.
    int64_t max_err = 0;
    if (!no_verify) {
        try {
            max_err = pla::verify_epsilon(result, keys);
        } catch (const std::exception& e) {
            std::cerr << "VERIFICATION FAILED: " << e.what() << "\n";
            return 1;
        }
    }

    // Emit JSONL to stdout.
    // Runner will capture and append to results/raw/pla_only.jsonl
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{"
        << "\"exp_id\":\""        << exp_id     << "\","
        << "\"scenario\":\"pla_only\","
        << "\"index\":\"none\","
        << "\"pla\":\""           << algo_s     << "\","
        << "\"epsilon\":"         << epsilon    << ","
        << "\"threads\":"         << threads    << ","
        << "\"dataset\":\""       << (dataset.empty() ? dist_s + "_synth" : dataset) << "\","
        << "\"workload\":\"build_only\","
        << "\"build_ms\":"        << build_ms   << ","
        << "\"seg_cnt\":"         << result.segments.size() << ","
        << "\"bytes_index\":"     << result.bytes()         << ","
        << "\"max_err\":"         << max_err    << ","
        << "\"n_keys\":"          << keys.size()<< ","
        << "\"dup_runs\":"        << result.dup_runs << ","
        << "\"ops_s\":0,\"p50_ns\":0,\"p95_ns\":0,\"p99_ns\":0,"
        << "\"cache_misses\":0,\"branches\":0,\"branch_misses\":0,"
        << "\"instructions\":0,\"cycles\":0,\"rss_mb\":0,"
        << "\"fetch_strategy\":-1,\"io_pages\":0"
        << "}\n";
    return 0;
}
