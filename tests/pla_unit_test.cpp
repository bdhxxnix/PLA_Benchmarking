// tests/pla_unit_test.cpp
// Unit tests for all three PLA algorithms.
// Verifies: |pred(keys[i]) - i| <= epsilon for all i.
// Exits 0 on pass, non-zero on failure.

#include <pla/pla_api.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

// ─── Test helpers ────────────────────────────────────────────────────────────
struct TestCase {
    std::string name;
    std::vector<uint64_t> keys;
};

static std::vector<uint64_t> make_sequential(size_t n) {
    std::vector<uint64_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = i * 2; // even numbers
    return v;
}

static std::vector<uint64_t> make_random_sorted(size_t n, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = rng();
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

static std::vector<uint64_t> make_duplicates(size_t n) {
    // n/4 unique values, each repeated 4 times.
    std::vector<uint64_t> v;
    v.reserve(n);
    for (size_t i = 0; i < n/4; ++i)
        for (int r = 0; r < 4; ++r)
            v.push_back(i * 100);
    std::sort(v.begin(), v.end());
    v.resize(n);
    return v;
}

static std::vector<uint64_t> make_lognormal(size_t n, uint64_t seed = 99) {
    std::mt19937_64 rng(seed);
    std::lognormal_distribution<double> dist(0.0, 2.0);
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = static_cast<uint64_t>(dist(rng) * 1e9);
    std::sort(v.begin(), v.end());
    return v;
}

// ─── Run one test case ────────────────────────────────────────────────────────
static bool run_test(const std::string& label,
                     const std::vector<uint64_t>& keys,
                     int64_t epsilon,
                     pla::PlaAlgo algo) {
    if (keys.empty()) {
        std::cerr << "[SKIP] " << label << " (empty keys)\n";
        return true;
    }

    pla::PlaOptions opts;
    opts.handle_duplicates = true;

    pla::PlaResult result = pla::build_pla(keys, epsilon, algo, opts);

    // Verify epsilon guarantee.
    int64_t max_err = 0;
    bool ok = true;
    for (size_t i = 0; i < keys.size(); ++i) {
        const pla::Segment* seg = result.find_segment(keys[i]);
        if (!seg) {
            std::cerr << "  no segment for keys[" << i << "]=" << keys[i] << "\n";
            ok = false;
            max_err = std::max(max_err, static_cast<int64_t>(i));
            continue;
        }
        int64_t pred = static_cast<int64_t>(seg->predict_raw(keys[i]));
        int64_t err  = std::abs(pred - static_cast<int64_t>(i));
        if (err > epsilon) {
            if (ok) // print first violation only
                std::cerr << "  VIOLATION at i=" << i << " key=" << keys[i]
                          << " pred=" << pred << " err=" << err << " eps=" << epsilon << "\n";
            ok = false;
        }
        max_err = std::max(max_err, err);
    }

    const char* status = ok ? "PASS" : "FAIL";
    std::cout << "[" << status << "] " << label
              << "  segs=" << result.segments.size()
              << "  max_err=" << max_err
              << "  build_ms=" << result.build_ms << "\n";
    return ok;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    bool all_pass = true;

    // Dataset sizes to test.
    for (size_t n : {1UL, 2UL, 100UL, 10000UL, 100000UL}) {
        for (int64_t eps : {1LL, 8LL, 32LL, 64LL, 128LL}) {
            for (auto algo : {pla::PlaAlgo::Optimal, pla::PlaAlgo::Swing, pla::PlaAlgo::Greedy}) {
                std::string algo_s = pla::algo_to_string(algo);
                std::string prefix = algo_s + "_n" + std::to_string(n) + "_e" + std::to_string(eps);

                all_pass &= run_test(prefix + "_seq",  make_sequential(n),     eps, algo);
                all_pass &= run_test(prefix + "_rnd",  make_random_sorted(n),  eps, algo);
                all_pass &= run_test(prefix + "_dup",  make_duplicates(n),     eps, algo);
                all_pass &= run_test(prefix + "_logn", make_lognormal(n),      eps, algo);
            }
        }
    }

    if (all_pass) {
        std::cout << "\nAll tests PASSED.\n";
        return 0;
    } else {
        std::cerr << "\nSome tests FAILED.\n";
        return 1;
    }
}
