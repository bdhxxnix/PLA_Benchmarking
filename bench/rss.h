#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <unistd.h>

static size_t get_rss_mb() {
#ifdef __linux__
    std::ifstream f("/proc/self/statm");
    if (!f) return 0;
    long total = 0, rss = 0;
    f >> total >> rss;
    if (f.fail()) return 0;
    long page_sz = sysconf(_SC_PAGESIZE);
    if (page_sz <= 0) page_sz = 4096;
    return static_cast<size_t>(rss) * static_cast<size_t>(page_sz) / (1024 * 1024);
#else
    return 0;
#endif
}
