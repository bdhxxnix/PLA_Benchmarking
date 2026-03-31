#!/usr/bin/env bash
# scripts/drop_caches.sh
# 刷写文件系统缓冲区并丢弃 OS 页缓存（需要 root 权限）。
# 用于磁盘基准测试前确保冷缓存状态。
#
# 用法: sudo bash scripts/drop_caches.sh [--sync-only]
#   --sync-only  只执行 sync，不写 drop_caches（适合非 root 环境）

set -euo pipefail

SYNC_ONLY=0
for arg in "$@"; do
    [[ "$arg" == "--sync-only" ]] && SYNC_ONLY=1
done

echo "[drop_caches] Syncing filesystem buffers..."
sync

if [[ "$SYNC_ONLY" == "1" ]]; then
    echo "[drop_caches] sync-only mode; page cache NOT dropped."
    exit 0
fi

if [[ "$EUID" -ne 0 ]]; then
    echo "[drop_caches] ERROR: root required to drop page cache."
    echo "  Run as: sudo bash scripts/drop_caches.sh"
    echo "  Or use --sync-only to just flush buffers."
    exit 1
fi

# 3 = 丢弃 pagecache + dentries + inodes
echo 3 > /proc/sys/vm/drop_caches
echo "[drop_caches] Page cache dropped (echo 3 > /proc/sys/vm/drop_caches)."
