#pragma once
// pla/include/pla/pla_api.h
// Unified PLA interface for pla-learned-index-bench.
//
// Semantics (PLABench §2):
//   Input : sorted keys[0..n-1] (uint64_t), implicit y = position index.
//   ε     : |predict(keys[i]) - i| <= ε  for all i.
//   Duplicates: handled via PGM-index canonical mapping (float→nextafter,
//               int→x+1 run compression, sentinel > last key → n).
//
// Three algorithms selectable at compile time via -DPLA_ALGO=optimal|swing|greedy
// or at runtime via PlaAlgo enum.

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace pla {

// ─── Segment representation ──────────────────────────────────────────────────
struct Segment {
    uint64_t key_lo;    // first key covered by this segment
    uint64_t key_hi;    // last key covered (inclusive), or UINT64_MAX for last segment
    double   slope;
    double   intercept;
    int64_t  rank_lo;   // position of key_lo in the original array
    int64_t  rank_hi;   // position of key_hi

    // Predict position for a query key (unclamped).
    [[nodiscard]] double predict_raw(uint64_t key) const noexcept {
        return slope * static_cast<double>(key - key_lo) + intercept;
    }
};

// ─── Search range ────────────────────────────────────────────────────────────
// Mirrors PGM-index ApproxPos semantics: lo = max(0, p-ε), hi = min(n, p+ε+2)
struct SearchRange {
    int64_t lo;
    int64_t hi;
};

inline SearchRange make_range(int64_t pred, int64_t epsilon, int64_t n) noexcept {
    return {
        std::max<int64_t>(0, pred - epsilon),
        std::min<int64_t>(n,  pred + epsilon + 2)
    };
}

// ─── Algorithm selector ──────────────────────────────────────────────────────
enum class PlaAlgo {
    Optimal,   // OptimalPLA via PGM-index make_segmentation[_par]
    Swing,     // SwingFilter (O(1) space, FITing-Tree pivot = segment start)
    Greedy,    // GreedyPLA   (same as Swing but pivot = midpoint of p1,p2)
};

inline PlaAlgo algo_from_string(const std::string& s) {
    if (s == "optimal") return PlaAlgo::Optimal;
    if (s == "swing")   return PlaAlgo::Swing;
    if (s == "greedy")  return PlaAlgo::Greedy;
    throw std::invalid_argument("Unknown PLA algorithm: " + s);
}

inline const char* algo_to_string(PlaAlgo a) {
    switch (a) {
        case PlaAlgo::Optimal: return "optimal";
        case PlaAlgo::Swing:   return "swing";
        case PlaAlgo::Greedy:  return "greedy";
    }
    return "unknown";
}

// ─── Build result ────────────────────────────────────────────────────────────
struct PlaResult {
    std::vector<Segment> segments;
    int64_t  epsilon;
    size_t   n_keys;
    PlaAlgo  algo;
    double   build_ms;   // wall-clock build time in milliseconds
    size_t   dup_runs;   // number of duplicate runs collapsed (0 if none)

    // Memory footprint of segment array in bytes.
    [[nodiscard]] size_t bytes() const noexcept {
        return segments.size() * sizeof(Segment);
    }

    // Search: find the segment covering `key` (binary search on key_lo).
    // Returns pointer to segment, or nullptr if key < segments[0].key_lo.
    [[nodiscard]] const Segment* find_segment(uint64_t key) const noexcept {
        if (segments.empty()) return nullptr;
        // upper_bound on key_lo, then step back
        auto it = std::upper_bound(
            segments.begin(), segments.end(), key,
            [](uint64_t k, const Segment& s){ return k < s.key_lo; });
        if (it == segments.begin()) return nullptr;
        return &*(--it);
    }

    // Convenience: predict + clamp to [0, n_keys).
    [[nodiscard]] SearchRange search_range(uint64_t key) const noexcept {
        const Segment* seg = find_segment(key);
        if (!seg) return {0, std::min<int64_t>(static_cast<int64_t>(n_keys), epsilon + 2)};
        auto pred = static_cast<int64_t>(seg->predict_raw(key));
        return make_range(pred, epsilon, static_cast<int64_t>(n_keys));
    }
};

// ─── Extra build options ──────────────────────────────────────────────────────
struct PlaOptions {
    bool     handle_duplicates = true;  // collapse duplicate runs (PGM-index style)
    unsigned threads           = 1;     // parallel build threads (Optimal only; others single-threaded)
    bool     verbose           = false;
};

// ─── Primary API ─────────────────────────────────────────────────────────────
// Forward declarations; implementations in alg_optimal.h / alg_swing.h / alg_greedy.h
// and dispatched in pla_api.cpp (or inline below).

PlaResult build_pla(
    const uint64_t* keys,
    size_t          n,
    int64_t         epsilon,
    PlaAlgo         algo    = PlaAlgo::Optimal,
    PlaOptions      options = {}
);

// Convenience overload for std::vector.
inline PlaResult build_pla(
    const std::vector<uint64_t>& keys,
    int64_t                      epsilon,
    PlaAlgo                      algo    = PlaAlgo::Optimal,
    PlaOptions                   options = {}
) {
    return build_pla(keys.data(), keys.size(), epsilon, algo, options);
}

// Verify error guarantee: returns max |pred - i| over all keys.
// Throws if > epsilon (use in unit tests).
int64_t verify_epsilon(const PlaResult& result, const uint64_t* keys, size_t n);

inline int64_t verify_epsilon(const PlaResult& result, const std::vector<uint64_t>& keys) {
    return verify_epsilon(result, keys.data(), keys.size());
}

} // namespace pla

// ─── Dispatch to selected algorithm ──────────────────────────────────────────
// Include algorithm implementations here so the header is self-contained
// (all are header-only).
#include "alg_optimal.h"
#include "alg_swing.h"
#include "alg_greedy.h"

namespace pla {

inline PlaResult build_pla(
    const uint64_t* keys,
    size_t          n,
    int64_t         epsilon,
    PlaAlgo         algo,
    PlaOptions      options
) {
    switch (algo) {
        case PlaAlgo::Optimal: return detail::build_optimal(keys, n, epsilon, options);
        case PlaAlgo::Swing:   return detail::build_swing  (keys, n, epsilon, options);
        case PlaAlgo::Greedy:  return detail::build_greedy (keys, n, epsilon, options);
    }
    throw std::invalid_argument("Unknown PLA algorithm");
}

inline int64_t verify_epsilon(const PlaResult& result, const uint64_t* keys, size_t n) {
    int64_t max_err = 0;
    for (size_t i = 0; i < n; ++i) {
        const Segment* seg = result.find_segment(keys[i]);
        if (!seg) {
            max_err = std::max(max_err, static_cast<int64_t>(i));
            continue;
        }
        // Use llround so that floating-point predictions within 0.5 of an integer
        // boundary are not rounded away from the true rank by C++ truncation.
        int64_t pred = static_cast<int64_t>(std::llround(seg->predict_raw(keys[i])));
        max_err = std::max(max_err, std::abs(pred - static_cast<int64_t>(i)));
    }
    if (max_err > result.epsilon) {
        throw std::runtime_error(
            "epsilon violation: max_err=" + std::to_string(max_err) +
            " > epsilon=" + std::to_string(result.epsilon));
    }
    return max_err;
}

} // namespace pla
