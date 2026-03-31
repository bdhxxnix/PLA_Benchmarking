#pragma once
// pla/include/pla/alg_optimal.h
// OptimalPLA: wraps PGM-index internal::make_segmentation[_par].
//
// References:
//   PGM-index: include/pgm/piecewise_linear_model.hpp
//              internal::make_segmentation(first, last, epsilon, out)
//              internal::make_segmentation_par(first, last, epsilon, nthreads, out)
//   Canonical segment → (key, slope, intercept) extraction mirrors
//   PGMIndex<K,epsilon>::build_level().

#include <chrono>
#include <cstdint>
#include <vector>
#include <algorithm>

// Pull in PGM-index internals.  The submodule must be initialised.
// We guard so that the header compiles even if PGM headers are absent
// (stub fallback used in that case).
#if __has_include(<pgm/piecewise_linear_model.hpp>)
#  include <pgm/piecewise_linear_model.hpp>
#  define HAVE_PGM 1
#else
#  define HAVE_PGM 0
#  pragma message("PGM-index headers not found; OptimalPLA will use stub implementation")
#endif

namespace pla::detail {

#if HAVE_PGM

// Helper: convert PGM canonical_segment to our Segment.
// PGM canonical_segment stores (first, slope, intercept) as floating-point
// where first is the first key of the segment.
template<typename CanonicalSegment>
inline Segment canonical_to_segment(
        const CanonicalSegment& cs,
        int64_t rank_lo,
        int64_t rank_hi,
        uint64_t key_hi_val) {
    // PGM stores slope and intercept such that: pos ≈ slope*(key-first)+intercept
    auto [cs_slope, cs_intercept] = cs.get_floating_point_segment(cs.get_first_x());
    Segment s;
    s.key_lo    = static_cast<uint64_t>(cs.get_first_x());
    s.key_hi    = key_hi_val;
    s.slope     = cs_slope;
    s.intercept = cs_intercept;
    s.rank_lo   = rank_lo;
    s.rank_hi   = rank_hi;
    return s;
}

inline PlaResult build_optimal(
        const uint64_t* keys,
        size_t          n,
        int64_t         epsilon,
        const PlaOptions& opts) {

    auto t0 = std::chrono::steady_clock::now();

    // Handle empty input.
    if (n == 0) {
        PlaResult r;
        r.epsilon = epsilon; r.n_keys = 0; r.algo = PlaAlgo::Optimal; r.build_ms = 0;
        return r;
    }

    // Duplicate key handling: PGM-index collapses duplicate runs by remapping
    // them to unique values (int: x+1 for each repeat; stored as separate
    // internal keys that all point to the same rank block).
    // We replicate this by building a deduplicated view with rank offsets.
    std::vector<uint64_t> dedup_keys;
    std::vector<size_t>   rank_map; // original index → dedup index
    dedup_keys.reserve(n);
    rank_map.reserve(n);

    if (opts.handle_duplicates) {
        uint64_t prev = std::numeric_limits<uint64_t>::max();
        uint64_t mapped = 0;
        for (size_t i = 0; i < n; ++i) {
            uint64_t k = keys[i];
            if (i == 0) {
                dedup_keys.push_back(k);
                mapped = k;
            } else if (k == prev) {
                // Duplicate: shift by 1 in the mapped space.
                mapped = dedup_keys.back() + 1;
                dedup_keys.push_back(mapped);
            } else {
                dedup_keys.push_back(k);
                mapped = k;
            }
            rank_map.push_back(dedup_keys.size() - 1);
            prev = k;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            dedup_keys.push_back(keys[i]);
            rank_map.push_back(i);
        }
    }
    // Sentinel: key > last, maps to n (PGM convention).
    dedup_keys.push_back(dedup_keys.back() + 1);

    // Collect canonical segments from PGM internals.
    using K = uint64_t;
    using PGMSeg = pgm::internal::OptimalPiecewiseLinearModel<K, size_t>;

    std::vector<pgm::internal::canonical_segment<K,size_t>> raw_segs;
    raw_segs.reserve(n / (epsilon * 2) + 2);

    auto out_fn = [&](auto cs){ raw_segs.push_back(cs); };

    if (opts.threads > 1) {
#ifdef _OPENMP
        pgm::internal::make_segmentation_par(
            dedup_keys.begin(), dedup_keys.end() - 1, // exclude sentinel
            static_cast<size_t>(epsilon),
            opts.threads,
            out_fn);
#else
        pgm::internal::make_segmentation(
            dedup_keys.begin(), dedup_keys.end() - 1,
            static_cast<size_t>(epsilon),
            out_fn);
#endif
    } else {
        pgm::internal::make_segmentation(
            dedup_keys.begin(), dedup_keys.end() - 1,
            static_cast<size_t>(epsilon),
            out_fn);
    }

    // Convert to our Segment representation.
    PlaResult result;
    result.segments.reserve(raw_segs.size());

    for (size_t si = 0; si < raw_segs.size(); ++si) {
        int64_t rlo = static_cast<int64_t>(si == 0 ? 0 :
            /* binary search for rank of first key of segment si */
            std::lower_bound(dedup_keys.begin(), dedup_keys.end(),
                raw_segs[si].get_first_x()) - dedup_keys.begin());
        int64_t rhi = (si + 1 < raw_segs.size())
            ? static_cast<int64_t>(std::lower_bound(dedup_keys.begin(), dedup_keys.end(),
                raw_segs[si+1].get_first_x()) - dedup_keys.begin()) - 1
            : static_cast<int64_t>(n) - 1;

        uint64_t key_hi = (si + 1 < raw_segs.size())
            ? static_cast<uint64_t>(raw_segs[si+1].get_first_x()) - 1
            : std::numeric_limits<uint64_t>::max();

        result.segments.push_back(
            canonical_to_segment(raw_segs[si], rlo, rhi, key_hi));
    }

    auto t1 = std::chrono::steady_clock::now();
    result.epsilon  = epsilon;
    result.n_keys   = n;
    result.algo     = PlaAlgo::Optimal;
    result.build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.dup_runs = opts.handle_duplicates ? (n - (dedup_keys.size() - 1)) : 0;
    return result;
}

#else // !HAVE_PGM — stub

inline PlaResult build_optimal(
        const uint64_t* /*keys*/,
        size_t          n,
        int64_t         epsilon,
        const PlaOptions& /*opts*/) {
    // Stub: single segment covering everything.
    PlaResult r;
    r.epsilon = epsilon; r.n_keys = n; r.algo = PlaAlgo::Optimal; r.build_ms = 0;
    if (n > 0) {
        Segment s; s.key_lo = 0; s.key_hi = UINT64_MAX;
        s.slope = 1.0; s.intercept = 0; s.rank_lo = 0; s.rank_hi = (int64_t)n-1;
        r.segments.push_back(s);
    }
    return r;
}

#endif // HAVE_PGM

} // namespace pla::detail
