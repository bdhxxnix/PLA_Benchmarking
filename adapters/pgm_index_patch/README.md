# adapters/pgm_index_patch/README.md
# PGM-index adapter patch

## What this patch does

PGM-index already contains the reference OptimalPLA implementation in
`include/pgm/piecewise_linear_model.hpp` as `internal::make_segmentation`
and `internal::make_segmentation_par`.  Our `alg_optimal.h` calls these
functions directly without modifying PGM-index source.

However, to **switch the PLA algorithm** used *inside* PGMIndex<> itself
(not just our standalone benchmarks), apply the patch below.

## Patch: swap PLA algorithm in pgm_index.hpp

```diff
--- a/include/pgm/pgm_index.hpp
+++ b/include/pgm/pgm_index.hpp
@@ -... +... @@
-    pgm::internal::make_segmentation_par(
+    // Replaced by pla-learned-index-bench adapter:
+    pla::detail::build_optimal_segments(   // or build_swing_segments / build_greedy_segments
```

The minimal replacement points are:

1. **`build_level()`** — replace the `make_segmentation_par` call and the
   `out_fun = [&](auto cs){ ... segments.emplace_back(cs) }` lambda with a
   call to `pla::build_pla(keys, epsilon, PLA_ALGO, opts)` and convert the
   returned `pla::PlaResult::segments` to PGM's `Segment` struct.

2. **`Segment` struct** — PGM's Segment stores `{key, slope, intercept}` as
   `float` + `int`.  Our `pla::Segment` uses `double`; cast accordingly.

## Applying the patch

```bash
bash scripts/apply_patches.sh
# or manually:
cd third_party/PGM-index
patch -p1 < ../../adapters/pgm_index_patch/pgm_index.patch
```

## Patch file

See `pgm_index.patch` in this directory.  Generated with:
```bash
cd third_party/PGM-index && git diff > ../../adapters/pgm_index_patch/pgm_index.patch
```
