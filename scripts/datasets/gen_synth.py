#!/usr/bin/env python3
# scripts/datasets/gen_synth.py
# 生成合成 uint64 数据集（已排序，一列，二进制格式）。
#
# 支持分布: uniform, normal, lognormal, zipf, mixture, duplicates
#
# 用法:
#   python3 scripts/datasets/gen_synth.py \
#       --dist lognormal --n 200000000 --output data/synth_lognormal_200M.bin
#   python3 scripts/datasets/gen_synth.py --dist uniform --n 1000000 --text

import argparse
import struct
import sys
import numpy as np
from pathlib import Path


def gen_uniform(n: int, seed: int = 42) -> np.ndarray:
    """均匀分布：[0, 2^64) 随机无符号整数，排序后去重。"""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
    v.sort()
    return np.unique(v)


def gen_normal(n: int, seed: int = 42) -> np.ndarray:
    """正态分布：均值 2^32, 标准差 2^28，截断到 [0, 2^64)，排序。"""
    rng = np.random.default_rng(seed)
    mu = 2**32
    sigma = 2**28
    v = rng.normal(mu, sigma, size=n)
    v = np.clip(v, 0, 2**63 - 1).astype(np.uint64)
    v.sort()
    return v


def gen_lognormal(n: int, seed: int = 42) -> np.ndarray:
    """对数正态分布：sigma=2.0，乘以 1e9 映射到 uint64 范围。"""
    rng = np.random.default_rng(seed)
    v = rng.lognormal(mean=0.0, sigma=2.0, size=n)
    v = (v * 1e9).astype(np.uint64)
    v.sort()
    return v


def gen_zipf(n: int, alpha: float = 1.1, seed: int = 42) -> np.ndarray:
    """Zipf 分布（幂律），排序后的 uint64 整数。"""
    rng = np.random.default_rng(seed)
    # numpy zipf 参数 a > 1；此处 alpha 为幂指数。
    a = alpha + 1
    raw = rng.zipf(a=a, size=n * 2)  # 多采以弥补截断
    raw = raw[:n].astype(np.uint64) * 100
    raw.sort()
    return raw


def gen_mixture(n: int, seed: int = 42) -> np.ndarray:
    """混合分布：50% uniform + 50% lognormal。"""
    half = n // 2
    u = gen_uniform(half, seed)
    l = gen_lognormal(n - half, seed + 1)
    v = np.concatenate([u, l]).astype(np.uint64)
    v.sort()
    return v


def gen_duplicates(n: int, unique_ratio: float = 0.25, seed: int = 42) -> np.ndarray:
    """含大量重复值：unique_ratio 比例的唯一值，每个重复 1/unique_ratio 次。"""
    rng = np.random.default_rng(seed)
    n_unique = max(1, int(n * unique_ratio))
    unique_vals = rng.integers(0, 2**32, size=n_unique, dtype=np.uint64)
    # 随机采样到 n 个（允许重复）。
    idx = rng.integers(0, n_unique, size=n)
    v = unique_vals[idx]
    v.sort()
    return v


GENERATORS = {
    "uniform":    gen_uniform,
    "normal":     gen_normal,
    "lognormal":  gen_lognormal,
    "zipf":       gen_zipf,
    "mixture":    gen_mixture,
    "duplicates": gen_duplicates,
}


def main():
    ap = argparse.ArgumentParser(description="生成合成 uint64 数据集")
    ap.add_argument("--dist",   required=True, choices=list(GENERATORS),
                    help="分布类型")
    ap.add_argument("--n",      type=int, default=1_000_000,
                    help="生成 key 数量（默认 1M）")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--output", default="",
                    help="输出文件路径（.bin 二进制 uint64）；空则写 stdout")
    ap.add_argument("--text",   action="store_true",
                    help="以文本格式输出（每行一个整数），忽略 --output")
    ap.add_argument("--zipf-alpha", type=float, default=1.1)
    args = ap.parse_args()

    gen_fn = GENERATORS[args.dist]
    kwargs: dict = {"seed": args.seed}
    if args.dist == "zipf":
        kwargs["alpha"] = args.zipf_alpha

    print(f"[gen_synth] dist={args.dist} n={args.n:,} seed={args.seed}",
          file=sys.stderr)
    keys = gen_fn(args.n, **kwargs)
    actual_n = len(keys)
    print(f"[gen_synth] generated {actual_n:,} keys (unique after sort)",
          file=sys.stderr)

    if args.text:
        for k in keys:
            sys.stdout.write(str(k) + "\n")
        return

    # 写二进制（小端 uint64）。
    out = args.output
    if not out:
        out = f"synth_{args.dist}_{actual_n // 1_000_000}M.bin"

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack(f"<{actual_n}Q", *keys))
    size_mb = Path(out).stat().st_size / 1024 / 1024
    print(f"[gen_synth] wrote {out}  ({size_mb:.1f} MiB)", file=sys.stderr)


if __name__ == "__main__":
    main()
