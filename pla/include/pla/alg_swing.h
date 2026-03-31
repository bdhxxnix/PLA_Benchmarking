#pragma once
// pla/include/pla/alg_swing.h
// SwingFilter PLA — O(1) working space per segment.
//
// Algorithm (PLABench §3.2 / FITing-Tree §3):
//   pivot = key_lo (first point of current segment).
//   For each new point (x_i, i):
//     slope_lo = (i - epsilon - intercept_at_pivot) / (x_i - x_pivot)
//     slope_hi = (i + epsilon - intercept_at_pivot) / (x_i - x_pivot)
//     Tighten [slope_lower, slope_upper] ∩ [slope_lo, slope_hi].
//     If intersection is empty → emit segment with slope = (slope_lower+slope_upper)/2,
//       start new segment with x_i as pivot.
//
// FITing-Tree reference:
//   src/piecewise_linear_model.h — get_lower_slope/get_upper_slope,
//   get_segment() returns midpoint slope.
// PLABench description confirms pivot = segment start (first point).

#include <chrono>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>

namespace pla::detail {

inline PlaResult build_swing(
        const uint64_t* keys,
        size_t          n,
        int64_t         epsilon,
        const PlaOptions& opts) {

    auto t0 = std::chrono::steady_clock::now();

    PlaResult result;
    result.epsilon = epsilon;
    result.n_keys  = n;
    result.algo    = PlaAlgo::Swing;

    if (n == 0) { result.build_ms = 0; return result; }
    if (n == 1) {
        Segment s;
        s.key_lo = keys[0]; s.key_hi = std::numeric_limits<uint64_t>::max();
        s.slope = 0; s.intercept = 0; s.rank_lo = 0; s.rank_hi = 0;
        result.segments.push_back(s);
        result.build_ms = 0;
        return result;
    }

    // State for current segment.
    double slope_lo = -std::numeric_limits<double>::infinity();
    double slope_hi =  std::numeric_limits<double>::infinity();

    // Pivot: first point of current segment.
    uint64_t x_pivot  = keys[0];
    int64_t  y_pivot  = 0;           // rank at pivot = rank_lo of segment
    int64_t  seg_start_rank = 0;

    auto emit = [&](int64_t last_rank, uint64_t last_key,
                    uint64_t next_key_lo) {
        Segment s;
        s.key_lo    = x_pivot;
        s.key_hi    = (next_key_lo == 0)
                        ? std::numeric_limits<uint64_t>::max()
                        : next_key_lo - 1;
        // Midpoint slope (FITing-Tree convention).
        s.slope     = std::isfinite(slope_lo) && std::isfinite(slope_hi)
                        ? (slope_lo + slope_hi) / 2.0
                        : 0.0;
        s.intercept = static_cast<double>(y_pivot);
        s.rank_lo   = seg_start_rank;
        s.rank_hi   = last_rank;
        result.segments.push_back(s);
    };

    for (size_t i = 1; i < n; ++i) {
        uint64_t xi = keys[i];
        int64_t  yi = static_cast<int64_t>(i);

        // Avoid division by zero (duplicate keys).
        if (xi == x_pivot) {
            // Duplicate: treat as same x → cannot change slope constraint;
            // if epsilon < |yi - y_pivot| we must emit.
            if (std::abs(yi - y_pivot) > epsilon) {
                // emit current segment ending at i-1, reset.
                emit(yi - 1, keys[i-1], xi);
                x_pivot = xi; y_pivot = yi; seg_start_rank = yi;
                slope_lo = -std::numeric_limits<double>::infinity();
                slope_hi =  std::numeric_limits<double>::infinity();
            }
            continue;
        }

        double dx      = static_cast<double>(xi - x_pivot);
        double new_slo = (static_cast<double>(yi) - epsilon - static_cast<double>(y_pivot)) / dx;
        double new_shi = (static_cast<double>(yi) + epsilon - static_cast<double>(y_pivot)) / dx;

        double merged_lo = std::max(slope_lo, new_slo);
        double merged_hi = std::min(slope_hi, new_shi);

        if (merged_lo > merged_hi) {
            // Intersection empty → emit segment, reset with current point as new pivot.
            emit(yi - 1, keys[i-1], xi);
            x_pivot = xi; y_pivot = yi; seg_start_rank = yi;
            slope_lo = -std::numeric_limits<double>::infinity();
            slope_hi =  std::numeric_limits<double>::infinity();
        } else {
            slope_lo = merged_lo;
            slope_hi = merged_hi;
        }
    }

    // Emit final segment.
    emit(static_cast<int64_t>(n) - 1, keys[n-1], 0 /* sentinel: next_key_lo unused */);

    auto t1 = std::chrono::steady_clock::now();
    result.build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.dup_runs = 0;
    return result;
}

} // namespace pla::detail
