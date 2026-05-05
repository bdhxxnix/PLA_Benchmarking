#pragma once

#include <cstddef>
#include <fstream>

static size_t get_rss_mb() {
#ifdef __linux__
    std::ifstream f("/proc/self/statm");
    if (!f) return 0;
    long total = 0, rss = 0;
    f >> total >> rss;
    if (f.fail()) return 0;
    return static_cast<size_t>(rss) * 4096 / (1024 * 1024);
#else
    return 0;
#endif
}
