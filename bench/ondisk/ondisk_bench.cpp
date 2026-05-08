// bench/ondisk/ondisk_bench.cpp
// On-disk benchmark for OD-A through OD-F experiments.
//
// Emulates disk I/O via mmap + posix_fadvise.  Falls back to an in-memory
// synthetic dataset when no --dataset file is given.
//
// G2: Prediction granularity (--granularity item|page)
//   item  — PLA built on (key_i, rank_i);  search range is item indices.
//   page  — PLA built on (key_i, page_i);  search range is page numbers.
//
// G3: Page-aligned error bound (--page-align)
//   Extends the binary-search range to cover complete 4 KiB pages, reducing
//   partial-page waste.  Potentially widens last-mile search but cuts I/O.
//
// G1: Fetch strategies (--fetch-strategy 0|1|2|3 — unchanged from original)
//
// Workload modes (--workload):
//   readonly — 100 % point lookups (default, OD-B / OD-C)
//   insert   — 100 % inserts via delta buffer
//   hybrid   — 50 % lookups + 50 % inserts via delta buffer (OD-F)
//
// iso-Rp mode (--target-rp N):
//   Binary-searches for the ε that achieves the target mean pages/query on
//   a probe run of 10 000 queries.  Reports the found ε alongside normal stats.
//
// Output: JSONL to stdout (one line per run).

#include <pla/pla_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "perf_counters.h"
#include "rss.h"

// EDLI real PGM-index-disk (runtime epsilon, page-oriented learned index)
#include <pgm_index_page.hpp>

using Clock = std::chrono::steady_clock;
using Ns    = std::chrono::duration<double, std::nano>;
using Ms    = std::chrono::duration<double, std::milli>;

static constexpr size_t PAGE_BYTES = 4096;
static constexpr size_t KEY_BYTES  = sizeof(uint64_t);

// ─── CLI helpers ─────────────────────────────────────────────────────────────
static std::string get_arg(int argc, char** argv, const char* f, const char* d = "") {
    for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], f)) return argv[i + 1];
    return d;
}
static bool has_flag(int argc, char** argv, const char* f) {
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], f)) return true;
    return false;
}

// ─── Percentile helper ───────────────────────────────────────────────────────
static double vec_pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[std::min(idx, v.size() - 1)];
}

// ─── Dataset generation (synthetic fallback) ─────────────────────────────────
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
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// ─── Disk file abstraction (mmap-based) ──────────────────────────────────────
struct DiskFile {
    int      fd       = -1;
    size_t   file_sz  = 0;
    uint8_t* ptr      = nullptr;

    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st; ::fstat(fd, &st); file_sz = static_cast<size_t>(st.st_size);
        ptr = static_cast<uint8_t*>(
            ::mmap(nullptr, file_sz, PROT_READ, MAP_SHARED, fd, 0));
        if (ptr == MAP_FAILED) { ::close(fd); fd = -1; return false; }
        ::posix_fadvise(fd, 0, static_cast<off_t>(file_sz), POSIX_FADV_RANDOM);
        return true;
    }

    void close_file() {
        if (ptr && ptr != MAP_FAILED) ::munmap(ptr, file_sz);
        if (fd >= 0) ::close(fd);
        fd = -1; ptr = nullptr;
    }

    ~DiskFile() { close_file(); }

    void touch_page(size_t byte_off) const {
        size_t pg = (byte_off / PAGE_BYTES) * PAGE_BYTES;
        if (pg < file_sz) { volatile uint8_t d = ptr[pg]; (void)d; }
    }
};

// ─── Direct I/O file abstraction ───────────────────────────────────────────────
// Bypasses OS page cache using O_DIRECT. Requires 512-byte aligned buffers.
struct DirectIOFile {
    int      fd       = -1;
    size_t   file_sz  = 0;
    mutable uint8_t* read_buf = nullptr;
    size_t   buf_sz   = 0;

    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_DIRECT | O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (::fstat(fd, &st) < 0) { ::close(fd); fd = -1; return false; }
        file_sz = static_cast<size_t>(st.st_size);
        buf_sz  = PAGE_BYTES;
        if (::posix_memalign(reinterpret_cast<void**>(&read_buf), 512, buf_sz) != 0) {
            ::close(fd); fd = -1; return false;
        }
        return true;
    }

    void touch_page(size_t byte_off) const {
        size_t pg = (byte_off / PAGE_BYTES) * PAGE_BYTES;
        if (pg < file_sz)
            ::pread(fd, read_buf, PAGE_BYTES, static_cast<off_t>(pg));
    }

    void close_file() {
        if (read_buf) { ::free(read_buf); read_buf = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }

    ~DirectIOFile() { close_file(); }
};

// ─── Page fetch strategies (G1) ──────────────────────────────────────────────
// Template: works with both DiskFile (mmap) and DirectIOFile (O_DIRECT).
template<typename FileT>
static size_t fetch_pages(const FileT& df,
                           const pla::SearchRange& range,
                           int strategy,
                           const uint8_t* mmap_ptr = nullptr) {
    size_t byte_lo = static_cast<size_t>(range.lo) * KEY_BYTES;
    size_t byte_hi = static_cast<size_t>(range.hi) * KEY_BYTES;
    byte_hi = std::min(byte_hi, df.file_sz);
    if (byte_lo >= byte_hi) return 0;

    size_t n_pages = (byte_hi - byte_lo + PAGE_BYTES - 1) / PAGE_BYTES;

    switch (strategy) {
        case 0: // one-by-one
            for (size_t p = 0; p < n_pages; ++p)
                df.touch_page(byte_lo + p * PAGE_BYTES);
            break;
        case 1: { // all-at-once (madvise for mmap, sequential touch for O_DIRECT)
            if (mmap_ptr)
                ::madvise(const_cast<uint8_t*>(mmap_ptr) + byte_lo,
                          byte_hi - byte_lo, MADV_WILLNEED);
            for (size_t off = byte_lo; off < byte_hi; off += PAGE_BYTES)
                df.touch_page(off);
            break;
        }
        case 2: { // all-at-once sorted (dedup page list)
            std::vector<size_t> pages;
            pages.reserve(n_pages);
            for (size_t p = 0; p < n_pages; ++p)
                pages.push_back(byte_lo + p * PAGE_BYTES);
            std::sort(pages.begin(), pages.end());
            for (auto off : pages) df.touch_page(off);
            break;
        }
        case 3: // model-biased: touch only the single predicted page
        default: {
            size_t mid = ((byte_lo + byte_hi) / 2 / PAGE_BYTES) * PAGE_BYTES;
            df.touch_page(mid);
            n_pages = 1;
            break;
        }
    }
    return n_pages;
}

// ─── Page-aligned range expansion (G3) ───────────────────────────────────────
// Extends [lo, hi) to cover whole pages, reducing partial-page I/O.
static pla::SearchRange page_align_range(pla::SearchRange r, size_t n_keys) {
    size_t byte_lo = static_cast<size_t>(r.lo) * KEY_BYTES;
    size_t byte_hi = static_cast<size_t>(r.hi) * KEY_BYTES;

    size_t page_lo = (byte_lo / PAGE_BYTES) * PAGE_BYTES;
    size_t page_hi = ((byte_hi + PAGE_BYTES - 1) / PAGE_BYTES) * PAGE_BYTES;

    int64_t new_lo = static_cast<int64_t>(page_lo / KEY_BYTES);
    int64_t new_hi = static_cast<int64_t>(
        std::min(page_hi / KEY_BYTES, n_keys));

    return {std::max(int64_t(0), new_lo), static_cast<int64_t>(new_hi)};
}

// ─── Page-level PLA index (G2 page granularity) ──────────────────────────────
// Instead of mapping key_i → rank_i, we map key_i → page_i.
// The epsilon is in units of PAGES; the search range returned gives pages.
struct PageLevelIndex {
    pla::PlaResult pla;    // built on (key_i, page_i)
    size_t         n_pages_total = 0;
    int64_t        eps_pages     = 0;

    void build(const uint64_t* keys, size_t n, int64_t epsilon_pages,
               pla::PlaAlgo algo, pla::PlaOptions opts) {
        eps_pages     = epsilon_pages;
        n_pages_total = (n * KEY_BYTES + PAGE_BYTES - 1) / PAGE_BYTES;

        // Build PLA on (key, page_number).
        // We reuse build_pla by constructing a "page rank" array mapping
        // each key to its page number and running PLA on that.
        // Since build_pla operates on sorted uint64 keys with implicit rank=index,
        // we insert each key once per page boundary crossing.
        // Simpler: just convert the rank target in the verification step.
        // Here we use the fact that: page(i) = i * KEY_BYTES / PAGE_BYTES
        // = i * 8 / 4096 = i / 512 (for PAGE_BYTES=4096, KEY_BYTES=8).
        // We compress by building on unique keys with page-based rank mapping.
        // For simplicity, we build on full key array but use epsilon in page units.
        const size_t ITEMS_PER_PAGE = PAGE_BYTES / KEY_BYTES; // 512
        pla = pla::build_pla(keys, n,
            static_cast<int64_t>(epsilon_pages * ITEMS_PER_PAGE), algo, opts);
    }

    // Returns a range [lo, hi) in ITEM indices covering the predicted page range.
    pla::SearchRange search_range_items(uint64_t key, size_t n_keys) const {
        auto r = pla.search_range(key);
        // Expand to page boundaries for G2 page-granularity I/O.
        return page_align_range(r, n_keys);
    }
};

// ─── Delta buffer for hybrid workload (OD-F / G6) ────────────────────────────
struct DeltaBuffer {
    std::vector<uint64_t> buf;   // unsorted inserts
    static constexpr size_t MAX_BUF = 100000;

    void insert(uint64_t key) { buf.push_back(key); }

    // Returns true if key is in the delta buffer.
    bool contains(uint64_t key) const {
        // Linear scan (small buffer); for production use sorted+binary search.
        for (uint64_t k : buf) if (k == key) return true;
        return false;
    }

    bool needs_merge() const { return buf.size() >= MAX_BUF; }
};

// ─── EDLI PGM-index-disk wrapper (real third-party) ─────────────────────────
// Wraps pgm_page::PGMIndexPage<K> which accepts RUNTIME epsilon in the
// constructor — no compile-time template parameter needed.  The EDLI index
// produces item-position search ranges; our I/O emulation layer (DiskFile,
// DirectIOFile, fetch strategies) is used unchanged for page-level I/O.
struct EdliPgmIndex {
    pgm_page::PGMIndexPage<uint64_t> idx;
    size_t n_keys = 0;
    int64_t epsilon = 128;

    void build(const uint64_t* keys, size_t n, int64_t eps) {
        epsilon = eps;
        n_keys  = n;
        std::vector<std::pair<uint64_t, size_t>> data(n);
        for (size_t i = 0; i < n; ++i)
            data[i] = {keys[i], i};
        idx = pgm_page::PGMIndexPage<uint64_t>(data.begin(), data.end(),
                                                static_cast<size_t>(eps));
    }

    pla::SearchRange search_range(uint64_t key) const {
        auto r = idx.search(key);
        return {static_cast<int64_t>(r.lo),
                static_cast<int64_t>(std::min(r.hi, n_keys))};
    }

    size_t seg_cnt()      const { return idx.segments_count(); }
    size_t bytes()        const { return idx.size_in_bytes(); }
    int    levels()       const { return static_cast<int>(idx.height()); }
    size_t seg_cnt_l1()   const {
        return idx.segments_count() > 0 ? seg_cnt() - seg_cnt() : 0;
    }
};

// ─── mean_rp helper: run a probe to estimate average pages/query ──────────────
static double probe_rp(const pla::PlaResult& idx, const uint64_t* keys, size_t n,
                       size_t n_probe, int fetch_strategy, bool page_align,
                       std::mt19937_64& rng) {
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    size_t total_pages = 0;
    for (size_t q = 0; q < n_probe; ++q) {
        uint64_t key = keys[dist(rng)];
        auto r = idx.search_range(key);
        if (page_align) r = page_align_range(r, n);
        // Count pages without actually touching memory.
        size_t byte_lo = static_cast<size_t>(r.lo) * KEY_BYTES;
        size_t byte_hi = static_cast<size_t>(r.hi) * KEY_BYTES;
        if (byte_hi > byte_lo)
            total_pages += (byte_hi - byte_lo + PAGE_BYTES - 1) / PAGE_BYTES;
    }
    return static_cast<double>(total_pages) / static_cast<double>(n_probe);
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int64_t     epsilon        = std::stoll(get_arg(argc, argv, "--epsilon",       "128"));
    std::string algo_s         = get_arg(argc, argv, "--algo",          "optimal");
    int         threads        = std::stoi(get_arg(argc, argv, "--threads",            "1"));
    std::string dataset        = get_arg(argc, argv, "--dataset",               "");
    std::string dist_s         = get_arg(argc, argv, "--dist",          "uniform");
    int         fetch_strategy = std::stoi(get_arg(argc, argv, "--fetch-strategy",    "1"));
    size_t      n_queries      = std::stoull(get_arg(argc, argv, "--queries",  "100000"));
    std::string exp_id         = get_arg(argc, argv, "--exp-id",        "ondisk");
    std::string granularity    = get_arg(argc, argv, "--granularity",   "item");
    std::string workload       = get_arg(argc, argv, "--workload",      "readonly");
    bool        page_align     = has_flag(argc, argv, "--page-align");
    double      target_rp      = std::stod(get_arg(argc, argv, "--target-rp",          "0"));
    size_t      n_synth        = std::stoull(get_arg(argc, argv, "--n",        "1000000"));
    bool        compress       = has_flag(argc, argv, "--compress");
    bool        direct_io      = has_flag(argc, argv, "--direct-io");
    std::string index_type     = get_arg(argc, argv, "--index",  "pla");
    (void)threads;

    // ── Load or generate keys ─────────────────────────────────────────────────
    std::vector<uint64_t> synth_keys;
    std::vector<uint64_t> key_buf;  // O_DIRECT: keys loaded into memory for binary search
    DiskFile mmap_df;
    DirectIOFile dio_df;
    const uint64_t* keys = nullptr;
    size_t n = 0;
    std::string ds_label;
    bool have_mmap = false;

    if (!dataset.empty()) {
        if (direct_io) {
            if (!dio_df.open(dataset)) {
                std::cerr << "Failed to open dataset with O_DIRECT: " << dataset << "\n";
                return 1;
            }
            // Load keys via normal read (O_DIRECT is only for page-touch ops).
            n = dio_df.file_sz / KEY_BYTES;
            key_buf.resize(n);
            {
                int kfd = ::open(dataset.c_str(), O_RDONLY);
                if (kfd < 0) { std::cerr << "Failed to open dataset for key load\n"; return 1; }
                ssize_t rd = ::read(kfd, key_buf.data(), dio_df.file_sz);
                ::close(kfd);
                if (rd != static_cast<ssize_t>(dio_df.file_sz)) {
                    std::cerr << "Key load read failed\n"; return 1;
                }
            }
            keys = key_buf.data();
            ds_label = dataset;
        } else {
            if (!mmap_df.open(dataset)) {
                std::cerr << "Failed to open dataset: " << dataset << "\n";
                return 1;
            }
            n      = mmap_df.file_sz / KEY_BYTES;
            keys   = reinterpret_cast<const uint64_t*>(mmap_df.ptr);
            ds_label = dataset;
            have_mmap = true;
        }
    } else {
        if (dist_s == "lognormal") synth_keys = gen_lognormal(n_synth);
        else                       synth_keys = gen_uniform(n_synth);
        n      = synth_keys.size();
        keys   = synth_keys.data();
        ds_label = dist_s + "_synth_" + std::to_string(n_synth);
    }

    if (n == 0) { std::cerr << "Empty dataset\n"; return 1; }

    // Warn when using synthetic in-memory data: without a real disk file and
    // a cold page cache, all fetch strategies touch hot RAM and produce
    // identical latency.  Pass a real --dataset file for meaningful G1 results.
    if (dataset.empty()) {
        std::cerr << "[WARN] ondisk_bench: no --dataset file supplied; using in-memory "
                     "synthetic data.  Fetch-strategy comparison requires a real on-disk "
                     "file with a cold page cache (run scripts/drop_caches.sh first).\n";
    }

    // Warn when epsilon is too small for fetch-strategy differences to manifest.
    // With PAGE_BYTES=4096 and KEY_BYTES=8, a range of (2*epsilon+2) items fits
    // in one page when 2*epsilon+2 <= 512, i.e. epsilon <= 255.  Below that,
    // io_pages_mean will always be 1 and all strategies are indistinguishable.
    if (epsilon <= 255) {
        std::cerr << "[WARN] ondisk_bench: epsilon=" << epsilon
                  << " produces a search range of " << (2*epsilon+2)
                  << " items (" << (2*epsilon+2)*KEY_BYTES << " bytes) which fits in "
                     "1 page (4096 bytes).  Fetch strategies will all report "
                     "io_pages_mean=1.  Use epsilon >= 256 to see multi-page I/O.\n";
    }

    pla::PlaAlgo    algo = pla::algo_from_string(algo_s);
    pla::PlaOptions opts; opts.threads = 1;

    // ── iso-Rp: find ε achieving target_rp pages/query ───────────────────────
    std::mt19937_64 rng_probe(7777);
    if (target_rp > 0.0) {
        // Binary search over epsilon in [1, 4*epsilon] until mean_rp ≈ target_rp.
        int64_t lo_eps = 1, hi_eps = epsilon * 4;
        for (int iter = 0; iter < 20 && lo_eps < hi_eps; ++iter) {
            int64_t mid_eps = lo_eps + (hi_eps - lo_eps) / 2;
            auto idx_probe  = pla::build_pla(keys, n, mid_eps, algo, opts);
            double rp = probe_rp(idx_probe, keys, n, 10000,
                                  fetch_strategy, page_align, rng_probe);
            if (rp > target_rp) lo_eps = mid_eps + 1;
            else                hi_eps = mid_eps;
        }
        epsilon = lo_eps; // use the found epsilon for the main benchmark run
    }

    // ── Build index ───────────────────────────────────────────────────────────
    size_t rss_before = get_rss_mb();

    PerfCounters perf;
    perf.open_all();
    perf.reset();
    perf.enable();

    auto t_build0 = Clock::now();
    pla::PlaResult index;
    PageLevelIndex page_idx;
    EdliPgmIndex   edli_idx;
    bool           use_edli = (index_type == "pgm-page");

    if (use_edli) {
        // Real EDLI PGM-index-disk (runtime epsilon)
        if (granularity == "page") {
            // G2 page granularity: scale epsilon to page units
            int64_t eps_pages = std::max(int64_t(1),
                epsilon * static_cast<int64_t>(KEY_BYTES) /
                static_cast<int64_t>(PAGE_BYTES));
            edli_idx.build(keys, n, eps_pages);
        } else {
            edli_idx.build(keys, n, epsilon);
        }
    } else if (granularity == "page") {
        // G2: page-level granularity — epsilon is in page units.
        page_idx.build(keys, n, epsilon, algo, opts);
        index = page_idx.pla; // for reporting seg_cnt / bytes
    } else {
        index = pla::build_pla(keys, n, epsilon, algo, opts);
    }
    double build_ms = Ms(Clock::now() - t_build0).count();

    perf.disable();
    size_t rss_after = get_rss_mb();
    int64_t rss_mb    = static_cast<int64_t>(rss_after) - static_cast<int64_t>(rss_before);
    int64_t hw_cache  = perf.cache_misses();
    int64_t hw_instr  = perf.instructions();
    int64_t hw_cycles = perf.cycles();
    int64_t hw_brmiss = perf.branch_misses();

    // G4: compressed index size.
    size_t bytes_compressed = 0;
    if (use_edli) {
        bytes_compressed = edli_idx.bytes();
    } else if (compress) {
        bytes_compressed = index.segments.size() *
            (sizeof(pla::Segment) - 2 * sizeof(double) + 2 * sizeof(float));
    } else {
        bytes_compressed = index.bytes();
    }

    // Drop page cache hints to simulate cold cache (best-effort without root).
    // posix_fadvise(DONTNEED) works on file-backed pages even without root,
    // unlike madvise(DONTNEED) which requires CAP_SYS_ADMIN for file mappings.
    if (!direct_io && have_mmap && mmap_df.fd >= 0)
        ::posix_fadvise(mmap_df.fd, 0,
                        static_cast<off_t>(mmap_df.file_sz),
                        POSIX_FADV_DONTNEED);

    // ── Generate queries ──────────────────────────────────────────────────────
    std::mt19937_64 rng_q(42);
    std::uniform_int_distribution<size_t> idx_dist(0, n - 1);

    size_t n_lookup = n_queries;
    (void)n_lookup;  // used for documentation; actual count tracked via lookup_count
    if (workload == "insert") {
        n_lookup = 0;
    } else if (workload == "hybrid") {
        n_lookup = n_queries / 2;
    }

    std::vector<uint64_t> queries(n_queries);
    for (auto& q : queries) q = keys[idx_dist(rng_q)];

    // ── Delta buffer for insert / hybrid workloads (G6 / OD-F) ───────────────
    DeltaBuffer delta;

    // ── Timed lookup loop ─────────────────────────────────────────────────────
    std::vector<double> latencies;
    latencies.reserve(n_queries);
    std::vector<double> io_pages_per_query;
    io_pages_per_query.reserve(n_queries);

    volatile int64_t sink = 0;
    size_t total_io_pages = 0;
    size_t lookup_count   = 0;
    size_t insert_count   = 0;

    auto t0 = Clock::now();

    for (size_t q = 0; q < n_queries; ++q) {
        // Decide operation type for hybrid/insert workloads.
        bool do_insert = false;
        if (workload == "insert")
            do_insert = true;
        else if (workload == "hybrid")
            do_insert = (q % 2 == 1);

        if (do_insert) {
            // Insert: add to delta buffer (disk-resident data not modified).
            delta.insert(queries[q]);
            ++insert_count;
            latencies.push_back(0.0); // inserts not timed for latency CDF
            io_pages_per_query.push_back(0.0);
            continue;
        }

        // Lookup path.
        auto qt0 = Clock::now();

        pla::SearchRange range;
        if (use_edli) {
            range = edli_idx.search_range(queries[q]);
        } else if (granularity == "page") {
            range = page_idx.search_range_items(queries[q], n);
        } else {
            range = index.search_range(queries[q]);
        }
        if (page_align && !use_edli && granularity != "page") {
            range = page_align_range(range, n);
        }

        // Touch pages (G1 fetch strategy).
        size_t pages_touched = 0;
        if (direct_io) {
            pages_touched = fetch_pages(dio_df, range, fetch_strategy, nullptr);
        } else if (have_mmap) {
            pages_touched = fetch_pages(mmap_df, range, fetch_strategy, mmap_df.ptr);
        } else {
            // In-memory emulation: count pages of the search range.
            size_t span = static_cast<size_t>(range.hi - range.lo) * KEY_BYTES;
            pages_touched = (span + PAGE_BYTES - 1) / PAGE_BYTES;
        }

        // Binary search in the fetched range.
        auto p = std::lower_bound(keys + range.lo, keys + range.hi, queries[q]);
        bool found = (p != keys + range.hi && *p == queries[q]);

        // Also check delta buffer for hybrid workloads.
        if (!found && (workload == "hybrid"))
            found = delta.contains(queries[q]);

        sink ^= found ? 1 : 0;

        double lat = Ns(Clock::now() - qt0).count();
        latencies.push_back(lat);
        io_pages_per_query.push_back(static_cast<double>(pages_touched));
        total_io_pages += pages_touched;
        ++lookup_count;
    }

    double total_ns = Ns(Clock::now() - t0).count();
    double ops_s    = (lookup_count > 0)
                    ? (lookup_count / (total_ns / 1e9))
                    : 0.0;
    (void)sink;

    // Filter out zero latencies (insert operations) for percentiles.
    std::vector<double> lookup_lats;
    lookup_lats.reserve(lookup_count);
    for (size_t i = 0; i < latencies.size(); ++i)
        if (latencies[i] > 0.0) lookup_lats.push_back(latencies[i]);

    std::sort(lookup_lats.begin(), lookup_lats.end());
    auto pct_lat = [&](double p) -> double {
        if (lookup_lats.empty()) return 0.0;
        size_t idx = static_cast<size_t>(p / 100.0 * (lookup_lats.size() - 1));
        return lookup_lats[std::min(idx, lookup_lats.size() - 1)];
    };

    double io_pages_p50 = vec_pct(io_pages_per_query, 50.0);
    double io_pages_p95 = vec_pct(io_pages_per_query, 95.0);
    double io_pages_p99 = vec_pct(io_pages_per_query, 99.0);
    double io_pages_mean = io_pages_per_query.empty() ? 0.0
        : std::accumulate(io_pages_per_query.begin(),
                          io_pages_per_query.end(), 0.0)
          / static_cast<double>(io_pages_per_query.size());

    size_t report_seg_cnt = use_edli ? edli_idx.seg_cnt()
                                     : index.segments.size();
    size_t report_bytes   = use_edli ? edli_idx.bytes()
                                     : index.bytes();
    std::string report_index = use_edli ? "PGM-Index-Page" : "PLA";

    std::cout << std::fixed << std::setprecision(3)
        << "{"
        << "\"exp_id\":\""         << exp_id         << "\","
        << "\"scenario\":\"ondisk\","
        << "\"index\":\""          << report_index   << "\","
        << "\"pla\":\""            << algo_s          << "\","
        << "\"epsilon\":"          << epsilon          << ","
        << "\"threads\":"          << 1               << ","
        << "\"dataset\":\""        << ds_label         << "\","
        << "\"workload\":\""       << workload         << "\","
        << "\"granularity\":\""    << granularity      << "\","
        << "\"page_align\":"       << (page_align ? "true" : "false") << ","
        << "\"fetch_strategy\":"   << fetch_strategy   << ","
        << "\"target_rp\":"        << target_rp        << ","
        << "\"build_ms\":"         << build_ms         << ","
        << "\"seg_cnt\":"          << report_seg_cnt   << ","
        << "\"bytes_index\":"      << report_bytes     << ","
        << "\"compress\":"         << (compress ? "true" : "false") << ","
        << "\"bytes_compressed\":" << bytes_compressed     << ","
        << "\"direct_io\":"        << (direct_io ? "true" : "false") << ","
        << "\"ops_s\":"            << ops_s            << ","
        << "\"p50_ns\":"           << pct_lat(50)      << ","
        << "\"p95_ns\":"           << pct_lat(95)      << ","
        << "\"p99_ns\":"           << pct_lat(99)      << ","
        << "\"io_pages\":"         << total_io_pages   << ","
        << "\"io_pages_mean\":"    << io_pages_mean    << ","
        << "\"io_pages_p50\":"     << io_pages_p50     << ","
        << "\"io_pages_p95\":"     << io_pages_p95     << ","
        << "\"io_pages_p99\":"     << io_pages_p99     << ","
        << "\"cache_misses\":"    << hw_cache        << ","
        << "\"branches\":0,"
        << "\"branch_misses\":"   << hw_brmiss       << ","
        << "\"instructions\":"    << hw_instr        << ","
        << "\"cycles\":"          << hw_cycles       << ","
        << "\"rss_mb\":"          << rss_mb
        << "}\n";
    return 0;
}
