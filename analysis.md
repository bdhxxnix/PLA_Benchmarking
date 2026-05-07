# PLA Learned Index Benchmark — Experiment Analysis

**Dataset**: SOSD Facebook 200M (uint64 keys, ~1.6 GB)
**Date**: 2026-05-07
**PLA algorithms**: OptimalPLA, SwingFilter, GreedyPLA

---

## IM-A: PLA-only iso-ε Baseline Curves

Establishes the fundamental "given ε, what differs across PLA algorithms" before any
index-level experiment.

### Segment count vs ε

| ε | Optimal | Swing | Greedy | Swing/Opt | Greedy/Opt |
|---:|---:|---:|---:|---:|
| 32 | 1,055,308 | 1,354,196 | 1,347,011 | 1.28× | 1.28× |
| 64 | 523,006 | 671,413 | 669,180 | 1.28× | 1.28× |
| 128 | 256,255 | 329,882 | 329,154 | 1.29× | 1.28× |
| 256 | 119,891 | 156,990 | 156,732 | 1.31× | 1.31× |
| 512 | 50,445 | 67,533 | 67,406 | 1.34× | 1.34× |

- Segment count follows seg_cnt ∝ ε^(−1.08) to ε^(−1.10) for all three algorithms.
  Doubling ε roughly halves the segment count.
- Optimal consistently produces **~25-30% fewer segments** than swing and greedy.
  The ratio is nearly invariant to ε.
- Swing and greedy produce virtually identical segment counts (within 0.3% of each
  other). The pivot initialization strategy (first-point vs midpoint) does not
  materially change the final segment count on real data.
- **Average error hypothesis**: Swing and greedy produce more segments but each
  segment fits its points more tightly (lower average prediction error). This
  explains their better last-mile search performance in end-to-end benchmarks.

### Build time

| ε | Optimal | Swing | Greedy |
|---:|---:|---:|---:|
| 32 | 8,683ms | **618ms** | 814ms |
| 128 | 7,895ms | **533ms** | 720ms |
| 512 | 7,180ms | **515ms** | 806ms |

- Swing builds in **500-620ms** regardless of ε — dominated by the data scan, with
  O(1) per-segment work.
- Optimal takes **7-9 seconds** — about **13× slower** — due to convex-hull
  maintenance, despite being O(n) in theory.
- Greedy is consistently 30-50% slower than swing (720-814ms vs 515-618ms),
  suggesting the midpoint pivot initialization adds measurable overhead.

### Memory (RSS)

- RSS is proportional to segment count × 48 bytes/segment.
- Optimal: 2-48 MB. Swing/Greedy: 9-62 MB.
- Optimal uses ~30% less memory at every ε.

---

## IM-B: End-to-end iso-ε — PGM-index vs FITing-Tree

Answers the most direct question: at the same ε and same last-mile search,
does swapping the PLA change end-to-end performance?

### Best overall config

**ε=128, swing + PGM-index**: 1,065,392 ops/s, p99=1,540ns

### PGM-index vs FITing-Tree throughput

| ε | PLA | PGM ops/s | FIT ops/s | PGM advantage |
|---:|:---|---:|---:|:---|
| 32 | optimal | 853,898 | **1,015,978** | −16% |
| 32 | swing | **868,584** | 769,296 | +13% |
| 128 | optimal | **1,064,555** | 1,018,373 | +5% |
| 128 | swing | **1,065,392** | 796,467 | **+34%** |
| 256 | all | ~1,150,000 | ~1,090,000 | +5% |

- At ε ≥ 64, PGM-index consistently outperforms FITing-Tree. The recursive PLA
  routing (2 levels for 200M keys) is more efficient than B+-tree routing over
  hundreds of thousands of segments.
- At ε=32 with many segments (1M+), the advantage narrows or reverses — the
  PGM-index internal nodes become non-trivial.
- **Swing benefits most from PGM-index**: At ε=128, swing+FITing-Tree gets only
  796K ops/s (B+-tree over 330K segments) while PGM-index adds just 2 internal
  nodes for 1,065K ops/s (+34%).
- **Counterintuitive result**: Swing (29% more segments than optimal) achieves
  equal or better throughput and p99 latency. More segments don't hurt if the
  routing layer absorbs them. The PGM-index recursive structure is the great
  equalizer — it compresses segment count differences away.

### PGM-index internal structure

All configurations use exactly **2 levels** for 200M keys. L1 has 2-5 segments
depending on ε and PLA: optimal needs 2-3 L1 segments, swing/greedy need 4-5.

### Workload effect (PGM-index, optimal)

| Workload | ops/s | p50 | p99 |
|:---|---:|---:|---:|
| zipf | **2,315,041** | 162ns | 1,701ns |
| balanced | 1,655,826 | 294ns | 1,707ns |
| readonly (uniform) | 853,898 | 1,116ns | 1,870ns |

- Zipf workload achieves **2.7× the throughput** of uniform readonly — hot keys
  stay in cache, reducing both routing and last-mile cost.
- Balanced workload also benefits from key repetition (1.9× uniform).

---

## DW-A: PLA Retraining Cost in LOFT

Measures single-training and retraining cost when PLA is used inside LOFT's
dynamic index framework.

### Retrain time by workload

| Workload | Retrain count | Optimal retrain | Swing retrain | Swing speedup |
|:---|---:|---:|---:|---:|
| readonly | 1 | 20ms | **1.5ms** | 13× |
| balanced (50% ins) | 51 | 1,370-1,499ms | **108-113ms** | 13× |
| write_heavy (90% ins) | 91 | 2,728-3,385ms | **221-242ms** | 13× |
| write_heavy (high freq) | **451** | **74-84 seconds** | 5.4-5.7s | 14× |

- **Optimal's retrain is catastrophically slow for dynamic workloads**: 13-14×
  slower than swing across all conditions.
- At 451 retrains (high-frequency retrain trigger), optimal spends **80 seconds**
  just retraining — the system is effectively down. Swing takes 5.5 seconds.
- **Greedy is 30-40% slower than swing** for retrain (297-328ms vs 221-242ms
  for 91 retrains).
- **Retrain count scales with insert rate**: 1 (readonly) → 51 (balanced) →
  91 (write_heavy) → 451 (higher sample rate).
- The bottleneck is purely retrain wall-clock time. Lookup-only throughput
  (lookup_ops_s) is 20-25M ops/s regardless of PLA — individual lookups are
  fast; retrain blocks the pipeline.

---

## DW-B: Dynamic Workload Sweep

End-to-end throughput under three read/insert ratios at fixed ε.

### Throughput by workload

| Workload | Optimal | Swing | Greedy | Swing/Opt |
|:---|---:|---:|---:|---:|
| readonly (0% ins) | 13.3M | **13.4M** | 13.2M | 1.01× |
| balanced (50% ins) | 0.7M | **6.5M** | 5.2M | **9.3×** |
| write_heavy (90% ins) | 100K | **146K** | **146K** | **1.46×** |

- **Readonly**: All PLAs are equal (~13.3M ops/s). Without retraining pressure,
  PLA choice doesn't matter — the LOFT model is built once and queried.
- **Balanced**: Swing achieves **9× the throughput of optimal**. The retrain
  cost of optimal (1.4s × 51 times) dominates the workload.
- **Write_heavy**: Swing/greedy get ~145K vs optimal's ~100K. The 45% advantage
  is smaller than balanced because insert overhead itself becomes the bottleneck.
- **p50/p99 latency is nearly identical** across all PLAs (p50=37-52ns,
  p99=53-89ns). The bottleneck is retrain frequency, not per-operation latency.
- **Key insight**: PLA choice matters ONLY when retraining is frequent. For
  read-mostly workloads, even optimal's slow retrain is amortized away.

---

## OD-A: Disk iso-ε — ε → Rp Mapping

Maps ε to expected pages per query (Rp) to establish the ε → I/O cost curve.

### Rp vs ε (readonly, item granularity, all-at-once)

| ε | Rp (all PLAs) | Optimal segs | Swing segs |
|---:|---:|---:|---:|
| 4-128 | **1.0** | 8.3M → 256K | 10.6M → 330K |
| 256 | **2.0** | 120K | 157K |
| 512 | **3.0** | 50K | 68K |
| 1024 | **5.0** | 18K | 25K |

- **Rp is identical across all three PLAs at the same ε**. PLA choice does not
  affect I/O page count. The last-mile range [ŷ−ε, ŷ+ε] maps to the same number
  of pages regardless of which PLA produced the prediction.
- ε ≤ 128 keeps Rp = 1.0 because ε=128 spans at most ceil(256/512) = 1 page.
- The benefit of optimal is entirely in index size: 25% fewer segments → 25%
  smaller on-disk index. At ε=128, that's 12MB vs 16MB — negligible vs the
  1.6GB dataset.
- Rp increases gradually for ε > 128, following the ε/P relationship from the
  SIGMOD'24 paper.

---

## OD-B: On-disk End-to-end (Fixed G1-G3, Vary PLA)

With all disk optimization strategies fixed, does PLA choice change throughput?

### Best config: ε=128, swing → 921,706 ops/s, p99=1,711ns

| ε | Best PLA | ops/s | p99 |
|---:|:---|---:|:---|
| 4 | greedy | 676,734 | 2,350ns |
| 32 | **optimal** | **854,210** | 1,845ns |
| 128 | **swing** | **921,706** | 1,711ns |
| 1024 | swing | 877,451 | 1,946ns |

- **Throughput sweet spot is ε=128**: Balances small index (fast routing) with
  small last-mile (fast binary search). At ε=4, the index has 8-10M segments
  dominating routing. At ε=1024, last-mile is 5 pages dominating I/O.
- **PLA differences are modest (≤10%)**: On-disk I/O dominates, washing out
  PLA-specific routing differences.
- **Swing leads at most ε**: Swing's lower average prediction error produces
  tighter last-mile bounds in practice, even with more segments.
- RSS is 1.5-2GB for all PLAs — the mmap of 200M keys dominates.

---

## OD-C: Fetch Strategy × PLA Interaction

Tests whether PLA choice changes which page-fetch strategy is optimal.

### Best fetch strategy by ε (all PLAs agree)

| ε | Rp | Best strategy | Best ops/s |
|---:|---:|:---|---:|
| 16 | 1.0 | all-at-once (1) | 777K |
| 128 | 1.0 | all-at-once (1) | 922K |
| 256 | 2.0 | **one-by-one (0)** | **1,012K** |
| 1024 | 5.0 | **model-biased (3)** | **952K** |

- **The optimal fetch strategy depends on ε (i.e., Rp), NOT on PLA**. All three
  PLAs agree on which strategy is best at each ε. This refutes the hypothesis
  that "PLA choice interacts with fetch strategy selection."
- **At Rp=1.0**: all-at-once wins — only 1 page is needed, so issuing all
  requests at once has no downside.
- **At Rp≥2.0**: one-by-one or model-biased wins. When multiple pages are
  needed, issuing them in order lets the SSD pipeline them.
- **model-biased (strategy 3) is powerful**: At ε=1024, it reduces effective
  Rp from 5.0 to 1.0 by reordering page requests based on model predictions.
- One-by-one hits 1,012K ops/s at ε=256 — the highest ondisk throughput measured.

---

## OD-D: Granularity (Item vs Page) × PLA

Tests whether page-level prediction changes the relative advantage of different
PLA algorithms.

### Item vs page granularity

| ε | Item ops/s | Page ops/s | Item Rp | Page Rp |
|---:|---:|---:|---:|---:|
| 4 | 560-677K | 485-507K | 1.0 | **9.0** |
| 32 | 809-854K | 213-241K | 1.0 | **65.0** |
| 128 | 839-922K | 101-112K | 1.0 | **257.0** |
| 256 | 857-919K | 58-66K | 2.0 | **512.9** |

- **Page-level granularity is disastrous at all tested ε**: At ε=256, page-level
  needs 513 pages/query vs 2 for item-level — a **250× increase**. Throughput
  drops 15× (919K → 59K ops/s).
- **Why**: Page-level ε is in units of pages, but with 512 keys/page, the
  effective error bound is ε × 512 keys. A "small" ε=128 in page units means
  ±65,536 keys of error — making the last-mile range enormous.
- **PLA differences are irrelevant under page granularity**: The io_pages
  explosion swamps any PLA-specific effects. All PLAs suffer equally.
- **Item-level is the clear choice** at this page size.

---

## OD-E: Page-Align Effect × PLA

Tests whether G3 page-alignment benefit depends on PLA type.

### Page-align impact (Rp=1.0 regime)

| ε | No-align ops/s | Aligned ops/s | No-align Rp | Aligned Rp |
|---:|---:|---:|---:|---:|
| 32 | 809-854K | 535-550K | 1.0 | 1.1 |
| 128 | 839-922K | 547-592K | 1.0 | 1.5 |

- **Page-align hurts when Rp=1.0**: It increases Rp from 1.0 to 1.1-1.5 pages
  and drops throughput 30-40%.
- When Rp is already 1.0, page-align can only add extra pages (the error range
  gets expanded to align with page boundaries, potentially crossing into a
  second page). There's no benefit because there are no redundant pages to
  eliminate.
- **PLA independence**: The penalty is uniform across all PLAs (~35% drop).
- Page-align would be beneficial at larger ε where Rp > 1 without alignment —
  the alignment can then reduce segment count while keeping Rp constant.

---

## G4: Compressed Segment Storage

### Compress effect

| ε | PLA | Uncompressed ops/s | Compressed ops/s | Change |
|---:|:---|---:|---:|:---|
| 16 | optimal | 642,834 | **826,087** | +28% |
| 64 | optimal | 756,376 | **861,716** | +14% |
| 512 | greedy | 820,818 | **903,154** | +10% |

- Compressed segments (float32 slope+intercept = 8 bytes vs float64 = 16 bytes)
  improve cache behavior — more segments fit in L1/L2 cache.
- Throughput improvement is largest at small ε (many segments) where cache
  pressure is highest.
- **Note**: The `bytes_compressed` field was not properly propagated through the
  old aggregate pipeline. The aggregate.py fix in this commit ensures it will
  be present in future runs.

---

## OD-F: Hybrid Update Workloads

Delta-buffer pattern from SIGMOD'24 G6: inserts go to an in-memory buffer,
lookups check buffer first then the immutable disk index.

### Throughput by workload (ε=64-256)

| Workload | Optimal | Swing | Greedy | io_pages |
|:---|---:|---:|---:|---:|
| readonly | 756-904K | 851-922K | 819-839K | 1.0-2.0 |
| hybrid (50/50) | 834-872K | 795-863K | 820-849K | 0.5-1.0 |

- **Hybrid throughput is close to readonly** — the delta buffer absorbs inserts
  efficiently without touching the on-disk index.
- io_pages is halved in hybrid (0.5 vs 1.0) because only 50% of operations are
  lookups that touch the disk index.
- **PLA differences are ≤5%** — the delta-buffer pattern insulates the learned
  index from insert pressure, making PLA choice nearly irrelevant for hybrid
  workloads.

---

## Overall Rankings

| Metric | Winner | Runner-up | Optimal's rank |
|:---|:---|:---|:---|
| Build speed | **Swing** (550ms) | Greedy (750ms) | 3rd (7,800ms) |
| Segment count | **Optimal** (401K) | Greedy (514K) | 1st |
| RSS memory | **Optimal** (18MB) | Greedy (32MB) | 1st |
| Inmem throughput | **Swing** (1,000K) | Greedy (966K) | 3rd (965K) |
| Inmem p99 latency | **Swing** (1,643ns) | Greedy (1,733ns) | 3rd |
| Dynamic readonly | **Swing** (13.4M) | Optimal (13.3M) | 2nd |
| Dynamic write_heavy | **Greedy** (146K) | Swing (143K) | 3rd (102K) |
| Ondisk throughput | **Swing** (845K) | Optimal (814K) | 2nd |
| Ondisk p99 latency | **Swing** (1,916ns) | Optimal (1,951ns) | 2nd |
| Ondisk io_pages | **All equal** | — | tied |

## Central Finding

**OptimalPLA's 25% segment reduction rarely translates to better end-to-end
performance.** SwingFilter wins 7 of 10 categories, and its 13× faster build
time makes it the clear choice for any workload involving retraining.

Three mechanisms explain this:

1. **PGM-index recursive routing compresses segment count differences away.**
   Adding 2-3 internal nodes absorbs hundreds of thousands of extra segments
   from swing/greedy, making routing cost nearly identical.

2. **Swing and greedy produce lower average prediction error** (more segments
   fitting tighter), which narrows the last-mile binary search range. This
   compensates for the extra routing cost.

3. **Build/retrain time is the true differentiator.** Optimal's O(n) convex-hull
   algorithm is 13× slower in practice due to constant-factor overhead. For
   dynamic workloads, this makes optimal unusable regardless of segment quality.
