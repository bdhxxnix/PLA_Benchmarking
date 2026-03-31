#!/usr/bin/env python3
# scripts/datasets/split_shards.py
# 按连续区间将排序好的 uint64 二进制文件切分为若干分片。
# 每个分片内部有序（继承自输入文件的全局有序性）。
#
# 用法:
#   python3 scripts/datasets/split_shards.py \
#       --input data/sosd_fb_200M.bin \
#       --shards 4 \
#       --output-dir data/shards/fb_200M

import argparse
import struct
import sys
from pathlib import Path


def load_binary(path: Path):
    """读取 uint64 小端二进制文件，返回 list[int]。"""
    size = path.stat().st_size
    n = size // 8
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack(f"<{n}Q", data))


def main():
    ap = argparse.ArgumentParser(description="按连续区间切分排序 uint64 数据集")
    ap.add_argument("--input",      required=True, help="输入 .bin 文件（排序好的 uint64）")
    ap.add_argument("--shards",     type=int, default=4, help="分片数量（默认 4）")
    ap.add_argument("--output-dir", required=True, help="输出目录")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[split_shards] Loading {in_path}...", file=sys.stderr)
    keys = load_binary(in_path)
    n    = len(keys)
    print(f"[split_shards] {n:,} keys → {args.shards} shards", file=sys.stderr)

    shard_size = (n + args.shards - 1) // args.shards
    stem = in_path.stem

    for i in range(args.shards):
        lo   = i * shard_size
        hi   = min(lo + shard_size, n)
        shard = keys[lo:hi]
        out_path = out_dir / f"{stem}_shard{i:03d}_of{args.shards:03d}.bin"
        with open(out_path, "wb") as f:
            f.write(struct.pack(f"<{len(shard)}Q", *shard))
        print(f"  shard {i:3d}: [{lo:,}, {hi:,})  n={hi-lo:,}  → {out_path.name}",
              file=sys.stderr)

    print("[split_shards] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
