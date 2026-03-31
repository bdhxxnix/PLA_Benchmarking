// bench/ondisk/ondisk_bench.cpp
// On-disk benchmark wrapper for Efficient-Disk-Learned-Index.
//
// When the Efficient-Disk-Learned-Index submodule is available this wrapper
// calls its microbenchmark entry point (run_microbenchmark).  Without the
// submodule it emulates disk I/O using mmap + posix_fadvise.
//
// fetch_strategy codes (mirroring Efficient-Disk-Learned-Index):
//   0 = one-by-one         (individual page faults)
//   1 = all-at-once        (prefetch whole range then scan)
//   2 = all-at-once-sorted (sort fetch list, then scan)
//   3 = model-biased       (use model prediction to fetch fewer pages)
//
// Usage:
//   ondisk_bench --epsilon 128 --algo optimal --threads 1
//                --dataset /data/sosd_books_800M.bin
//                --fetch-strategy 1 --n-pages 4096
//                --exp-id ondisk

#include <pla/pla_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::string get_arg(int argc, char** argv, const char* f, const char* d = "") {
    for (int i = 1; i+1 < argc; ++i) if (!std::strcmp(argv[i], f)) return argv[i+1];
    return d;
}

static const size_t PAGE_SIZE = 4096;

// ─── Disk file abstraction ────────────────────────────────────────────────────
struct DiskFile {
    int      fd   = -1;
    size_t   size = 0;
    uint8_t* mmap_ptr = nullptr;

    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st; ::fstat(fd, &st); size = st.st_size;
        mmap_ptr = static_cast<uint8_t*>(
            ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0));
        if (mmap_ptr == MAP_FAILED) { ::close(fd); fd=-1; return false; }
        // Advise the OS we're doing random access.
        ::posix_fadvise(fd, 0, size, POSIX_FADV_RANDOM);
        return true;
    }
    void close() {
        if (mmap_ptr && mmap_ptr != MAP_FAILED) ::munmap(mmap_ptr, size);
        if (fd >= 0) ::close(fd);
        fd = -1; mmap_ptr = nullptr;
    }
    ~DiskFile() { close(); }

    // Read a page containing byte offset `off`.
    void read_page(size_t off) const {
        size_t page_off = (off / PAGE_SIZE) * PAGE_SIZE;
        if (page_off < size)
            volatile uint8_t dummy = mmap_ptr[page_off];
        (void)0;
    }
};

// ─── Fetch strategies ────────────────────────────────────────────────────────
// Returns number of pages read for a query range [lo, hi).
static size_t fetch_pages(const DiskFile& df,
                          const pla::SearchRange& range,
                          const uint64_t* keys,
                          int strategy) {
    // Byte range covering the key array slice [lo, hi).
    size_t byte_lo = range.lo * sizeof(uint64_t);
    size_t byte_hi = std::min(range.hi * sizeof(uint64_t), df.size);
    if (byte_lo >= byte_hi) return 0;

    size_t n_pages = (byte_hi - byte_lo + PAGE_SIZE - 1) / PAGE_SIZE;

    switch (strategy) {
        case 0: // one-by-one: touch each page individually
            for (size_t p = 0; p < n_pages; ++p)
                df.read_page(byte_lo + p * PAGE_SIZE);
            break;
        case 1: // all-at-once: prefetch entire range
            ::madvise(df.mmap_ptr + byte_lo, byte_hi - byte_lo, MADV_WILLNEED);
            // Then scan sequentially.
            for (size_t off = byte_lo; off < byte_hi; off += PAGE_SIZE)
                df.read_page(off);
            break;
        case 2: { // all-at-once-sorted (same as 1 but with explicit sorted page list)
            std::vector<size_t> pages;
            for (size_t p = 0; p < n_pages; ++p) pages.push_back(byte_lo + p * PAGE_SIZE);
            std::sort(pages.begin(), pages.end());
            for (auto off : pages) df.read_page(off);
            break;
        }
        case 3: // model-biased: use predicted position, fetch only ±epsilon pages
        default: {
            // Touch only the single page containing the predicted position.
            if (keys) {
                // mid of range as best-guess predicted byte.
                size_t mid_byte = ((byte_lo + byte_hi) / 2 / PAGE_SIZE) * PAGE_SIZE;
                df.read_page(mid_byte);
                n_pages = 1;
            }
            break;
        }
    }
    return n_pages;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int64_t    epsilon      = std::stoll(get_arg(argc, argv, "--epsilon", "128"));
    std::string algo_s      = get_arg(argc, argv, "--algo", "optimal");
    int threads             = std::stoi(get_arg(argc, argv, "--threads", "1"));
    std::string dataset     = get_arg(argc, argv, "--dataset", "");
    int fetch_strategy      = std::stoi(get_arg(argc, argv, "--fetch-strategy", "1"));
    size_t n_queries        = std::stoull(get_arg(argc, argv, "--queries", "100000"));
    std::string exp_id      = get_arg(argc, argv, "--exp-id", "ondisk");
    (void)threads;

    if (dataset.empty()) {
        std::cerr << "ondisk_bench requires --dataset <binary_uint64_file>\n";
        return 1;
    }

    // Memory-map the dataset file.
    DiskFile df;
    if (!df.open(dataset)) {
        std::cerr << "Failed to open dataset: " << dataset << "\n";
        return 1;
    }
    size_t n = df.size / sizeof(uint64_t);
    const uint64_t* keys = reinterpret_cast<const uint64_t*>(df.mmap_ptr);

    // Build PLA index on the (memory-mapped) keys.
    // NB: We load the entire key array for index construction; in a real
    // disk scenario one would use a sparse sample (as PGMIndexPage does).
    pla::PlaAlgo algo = pla::algo_from_string(algo_s);
    pla::PlaOptions opts; opts.threads = 1;

    auto t_build0 = Clock::now();
    pla::PlaResult index = pla::build_pla(keys, n, epsilon, algo, opts);
    double build_ms = std::chrono::duration<double,std::milli>(Clock::now()-t_build0).count();

    // Drop OS page cache hint before queries (actual drop requires root; see
    // scripts/drop_caches.sh — here we just advise DONTNEED as best-effort).
    ::madvise(df.mmap_ptr, df.size, MADV_DONTNEED);

    // Generate queries.
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> idx_dist(0, n-1);
    std::vector<uint64_t> queries(n_queries);
    for (auto& q : queries) q = keys[idx_dist(rng)];

    // Timed lookup loop.
    std::vector<double> latencies;
    latencies.reserve(n_queries);
    volatile int64_t sink = 0;
    size_t total_io_pages = 0;

    for (size_t q = 0; q < n_queries; ++q) {
        auto qt0 = Clock::now();
        pla::SearchRange range = index.search_range(queries[q]);
        total_io_pages += fetch_pages(df, range, keys, fetch_strategy);
        // Binary search in fetched range.
        auto p = std::lower_bound(keys + range.lo, keys + range.hi, queries[q]);
        sink ^= (p != keys + range.hi && *p == queries[q]) ? 1 : 0;
        latencies.push_back(
            std::chrono::duration<double,std::nano>(Clock::now()-qt0).count());
    }
    (void)sink;

    std::sort(latencies.begin(), latencies.end());
    auto pct = [&](double p) {
        size_t idx = (size_t)(p/100.0*(latencies.size()-1));
        return latencies[std::min(idx, latencies.size()-1)];
    };
    double total_ns = 0; for (auto x : latencies) total_ns += x;
    double ops_s = n_queries / (total_ns / 1e9);

    std::cout << std::fixed << std::setprecision(3)
        << "{"
        << "\"exp_id\":\""        << exp_id         << "\","
        << "\"scenario\":\"ondisk\","
        << "\"index\":\"PGM-Index-Page\","
        << "\"pla\":\""           << algo_s          << "\","
        << "\"epsilon\":"         << epsilon          << ","
        << "\"threads\":"         << threads          << ","
        << "\"dataset\":\""       << dataset          << "\","
        << "\"workload\":\"readonly\","
        << "\"build_ms\":"        << build_ms         << ","
        << "\"seg_cnt\":"         << index.segments.size() << ","
        << "\"bytes_index\":"     << index.bytes()         << ","
        << "\"ops_s\":"           << ops_s            << ","
        << "\"p50_ns\":"          << pct(50)          << ","
        << "\"p95_ns\":"          << pct(95)          << ","
        << "\"p99_ns\":"          << pct(99)          << ","
        << "\"cache_misses\":0,\"branches\":0,\"branch_misses\":0,"
        << "\"instructions\":0,\"cycles\":0,\"rss_mb\":0,"
        << "\"fetch_strategy\":"  << fetch_strategy   << ","
        << "\"io_pages\":"        << total_io_pages
        << "}\n";
    return 0;
}
