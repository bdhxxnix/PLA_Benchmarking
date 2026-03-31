#!/usr/bin/env python3
"""
tools/runner/schema.py
Pydantic-free schema / dataclass definitions for experiment configs and results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


@dataclass
class ExperimentMatrix:
    """Parsed from configs/*.yaml – defines the cartesian product to run."""
    exp_name:        str
    scenario:        List[str]
    pla:             List[str]
    epsilon:         List[int]
    threads:         List[int]
    dataset:         List[str]
    workload:        List[str]
    fetch_strategy:  List[int]

    # Optional overrides
    n_keys:          int          = 1_000_000  # used for synthetic datasets
    queries:         int          = 1_000_000
    extra_flags:     Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentCase:
    """One cell in the expanded matrix."""
    exp_id:         str
    scenario:       str
    pla:            str
    epsilon:        int
    threads:        int
    dataset:        str
    workload:       str
    fetch_strategy: int
    n_keys:         int
    queries:        int
    extra_flags:    Dict[str, Any]

    def label(self) -> str:
        return (f"{self.scenario}_{self.pla}_e{self.epsilon}"
                f"_t{self.threads}_{self.dataset}_{self.workload}"
                f"_fs{self.fetch_strategy}")


@dataclass
class RunResult:
    """Raw metrics emitted by a benchmark binary (one JSONL line)."""
    exp_id:         str
    scenario:       str
    index:          str
    pla:            str
    epsilon:        int
    threads:        int
    dataset:        str
    workload:       str
    build_ms:       float
    seg_cnt:        int
    bytes_index:    int
    ops_s:          float
    p50_ns:         float
    p95_ns:         float
    p99_ns:         float
    cache_misses:   int
    branches:       int
    branch_misses:  int
    instructions:   int
    cycles:         int
    rss_mb:         float
    fetch_strategy: int
    io_pages:       int

    # Optional extras
    max_err:        Optional[int]   = None
    retrain_ms:     Optional[float] = None
    retrain_count:  Optional[int]   = None
    n_keys:         Optional[int]   = None
    dup_runs:       Optional[int]   = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        # Fill missing optional fields with None.
        for fname, fobj in cls.__dataclass_fields__.items():
            if fname not in filtered:
                filtered[fname] = None
        return cls(**filtered)
