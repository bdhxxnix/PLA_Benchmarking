#pragma once

// Shared perf_event_open HW counter wrapper for Linux.
// Counts cache-misses, instructions, cycles, branch-misses.
// Non-Linux: no-op stubs returning 0.

#ifdef __linux__
#  include <linux/perf_event.h>
#  include <sys/ioctl.h>
#  include <sys/syscall.h>
#  include <unistd.h>

static long perf_event_open_wrap(struct perf_event_attr* hw_event, pid_t pid,
                                  int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct PerfCounters {
    int fd_cache  = -1;
    int fd_instr  = -1;
    int fd_cycles = -1;
    int fd_brmiss = -1;

    void try_open(int& fd, uint32_t type, uint64_t config) {
        struct perf_event_attr pe = {};
        pe.type          = type;
        pe.size          = sizeof(pe);
        pe.config        = config;
        pe.disabled      = 1;
        pe.exclude_kernel= 1;
        pe.exclude_hv    = 1;
        fd = static_cast<int>(perf_event_open_wrap(&pe, 0, -1, -1, 0));
    }

    void open_all() {
        try_open(fd_cache,  PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
        try_open(fd_instr,  PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        try_open(fd_cycles, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
        try_open(fd_brmiss, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    }

    void ctrl(int cmd) {
        if (fd_cache  >= 0) ioctl(fd_cache,  cmd, 0);
        if (fd_instr  >= 0) ioctl(fd_instr,  cmd, 0);
        if (fd_cycles >= 0) ioctl(fd_cycles, cmd, 0);
        if (fd_brmiss >= 0) ioctl(fd_brmiss, cmd, 0);
    }

    void enable()  { ctrl(PERF_EVENT_IOC_ENABLE); }
    void reset()   { ctrl(PERF_EVENT_IOC_RESET);  }
    void disable() { ctrl(PERF_EVENT_IOC_DISABLE);}

    int64_t read_fd(int fd) const {
        if (fd < 0) return 0;
        int64_t val = 0;
        if (::read(fd, &val, sizeof(val)) != static_cast<ssize_t>(sizeof(val)))
            return 0;
        return val;
    }

    int64_t cache_misses()  const { return read_fd(fd_cache);  }
    int64_t instructions()  const { return read_fd(fd_instr);  }
    int64_t cycles()        const { return read_fd(fd_cycles); }
    int64_t branch_misses() const { return read_fd(fd_brmiss); }

    ~PerfCounters() {
        if (fd_cache  >= 0) close(fd_cache);
        if (fd_instr  >= 0) close(fd_instr);
        if (fd_cycles >= 0) close(fd_cycles);
        if (fd_brmiss >= 0) close(fd_brmiss);
    }
};
#else
struct PerfCounters {
    void open_all()  {}
    void enable()    {}
    void reset()     {}
    void disable()   {}
    int64_t cache_misses()  const { return 0; }
    int64_t instructions()  const { return 0; }
    int64_t cycles()        const { return 0; }
    int64_t branch_misses() const { return 0; }
};
#endif
