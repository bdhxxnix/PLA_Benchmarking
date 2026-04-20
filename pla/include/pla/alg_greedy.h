#pragma once
// pla/include/pla/alg_greedy.h
// GreedyPLA — identical slope-range logic to SwingFilter, but pivot = midpoint
// of the first two points of the current segment (PLABench §3.3).
//
// When only one point has been accumulated the pivot is that single point
// (degenerate case identical to SwingFilter). Once a second point arrives,
// pivot shifts to the midpoint of (x0, y0) and (x1, y1).

#include <chrono>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>

namespace pla::detail {

inline PlaResult build_greedy(
        const uint64_t* keys,
        size_t          n,
        int64_t         epsilon,
        const PlaOptions& opts) {

    auto t0 = std::chrono::steady_clock::now();

    PlaResult result;
    result.epsilon = epsilon;
    result.n_keys  = n;
    result.algo    = PlaAlgo::Greedy;

    if (n == 0) { result.build_ms = 0; return result; }
    if (n == 1) {
        Segment s;
        s.key_lo = keys[0]; s.key_hi = std::numeric_limits<uint64_t>::max();
        s.slope = 0; s.intercept = 0; s.rank_lo = 0; s.rank_hi = 0;
        result.segments.push_back(s);
        result.build_ms = 0;
        return result;
    }

    // Current segment state.
    // p0, p1 = first two points encountered after last reset.
    // After p1 is known, pivot = midpoint((x0,y0),(x1,y1)).
    uint64_t x0 = 0, x1 = 0;
    double   y0 = 0, y1 = 0;
    bool     have_p1 = false;

    // "Virtual pivot" used for slope-range calculation.
    // Before p1: pivot = p0.
    // After  p1: pivot_x = (x0+x1)/2.0, pivot_y = (y0+y1)/2.0
    double pivot_x = 0, pivot_y = 0;

    double slope_lo = -std::numeric_limits<double>::infinity();
    double slope_hi =  std::numeric_limits<double>::infinity();
    int64_t seg_start_rank = 0;

    auto emit = [&](int64_t last_rank, uint64_t next_key_lo) {
        Segment s;
        s.key_lo    = static_cast<uint64_t>(x0);
        s.key_hi    = (next_key_lo == 0)
                        ? std::numeric_limits<uint64_t>::max()
                        : next_key_lo - 1;
        double mid_slope = std::isfinite(slope_lo) && std::isfinite(slope_hi)
                        ? (slope_lo + slope_hi) / 2.0
                        : 0.0;
        s.slope     = mid_slope;
        // intercept such that slope*(x - key_lo) + intercept = predicted rank
        // = slope*(x - pivot_x) + pivot_y evaluated at x=key_lo gives the
        // intercept: slope*(x0 - pivot_x) + pivot_y
        s.intercept = mid_slope * (static_cast<double>(x0) - pivot_x) + pivot_y;
        s.rank_lo   = seg_start_rank;
        s.rank_hi   = last_rank;
        result.segments.push_back(s);
    };

    auto reset = [&](uint64_t xi, int64_t yi) {
        x0 = xi; y0 = static_cast<double>(yi);
        pivot_x = static_cast<double>(xi);
        pivot_y = static_cast<double>(yi);
        have_p1 = false;
        slope_lo = -std::numeric_limits<double>::infinity();
        slope_hi =  std::numeric_limits<double>::infinity();
        seg_start_rank = yi;
    };

    reset(keys[0], 0);

    for (size_t i = 1; i < n; ++i) {
        uint64_t xi = keys[i];
        double   yi = static_cast<double>(i);

        // Establish p1 (second point of segment) → shift pivot to midpoint.
        if (!have_p1) {
            if (xi == x0) {
                // Duplicate of first key; just absorb (epsilon check).
                if (std::abs(yi - y0) > static_cast<double>(epsilon)) {
                    emit(static_cast<int64_t>(i) - 1, xi);
                    reset(xi, static_cast<int64_t>(i));
                }
                continue;
            }
            // Second distinct key: set pivot = midpoint.
            x1 = xi; y1 = yi;
            pivot_x = (static_cast<double>(x0) + static_cast<double>(x1)) / 2.0;
            pivot_y = (y0 + y1) / 2.0;
            have_p1 = true;

            // Re-check both p0 and p1 against pivot for initial slope range.
            slope_lo = -std::numeric_limits<double>::infinity();
            slope_hi =  std::numeric_limits<double>::infinity();

            for (int pp = 0; pp <= 1; ++pp) {
                double xp = (pp == 0) ? static_cast<double>(x0) : static_cast<double>(x1);
                double yp = (pp == 0) ? y0 : y1;
                double dx = xp - pivot_x;
                if (std::abs(dx) < 1e-9) continue;
                double slo = (yp - epsilon - pivot_y) / dx;
                double shi = (yp + epsilon - pivot_y) / dx;
                if (dx < 0) std::swap(slo, shi);
                slope_lo = std::max(slope_lo, slo);
                slope_hi = std::min(slope_hi, shi);
            }
            if (slope_lo > slope_hi) {
                // Shouldn't normally happen with ε>=1 and 2 distinct keys, but handle.
                emit(static_cast<int64_t>(i) - 1, xi);
                reset(xi, static_cast<int64_t>(i));
            }
            continue;
        }

        // Normal case: update slope range with (xi, yi) relative to pivot.
        double dx = static_cast<double>(xi) - pivot_x;
        if (std::abs(dx) < 1e-9) {
            // xi very close to pivot_x (shouldn't happen for integral keys > x1).
            continue;
        }
        double new_slo = (yi - static_cast<double>(epsilon) - pivot_y) / dx;
        double new_shi = (yi + static_cast<double>(epsilon) - pivot_y) / dx;
        if (dx < 0) std::swap(new_slo, new_shi);

        double merged_lo = std::max(slope_lo, new_slo);
        double merged_hi = std::min(slope_hi, new_shi);

        if (merged_lo > merged_hi) {
            emit(static_cast<int64_t>(i) - 1, xi);
            reset(xi, static_cast<int64_t>(i));
        } else {
            slope_lo = merged_lo;
            slope_hi = merged_hi;
        }
    }

    // Emit last segment.
    emit(static_cast<int64_t>(n) - 1, 0);

    auto t1 = std::chrono::steady_clock::now();
    result.build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.dup_runs = 0;
    return result;
}

} // namespace pla::detail
