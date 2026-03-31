#!/usr/bin/env bash
# scripts/apply_patches.sh
# 将 adapters/ 下的补丁应用到各子模块。
# 仅当子模块已初始化（目录非空）时才尝试打补丁。
# 幂等：补丁已应用时跳过（patch --dry-run 检测）。

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

apply_patch() {
    local submodule_dir="$1"
    local patch_file="$2"
    local label="$3"

    if [ ! -d "$submodule_dir" ] || [ -z "$(ls -A "$submodule_dir" 2>/dev/null)" ]; then
        echo "  [SKIP] $label — submodule not initialised ($submodule_dir)"
        return
    fi
    if [ ! -f "$patch_file" ]; then
        echo "  [SKIP] $label — patch file not found ($patch_file)"
        return
    fi

    # 检测是否已经打过补丁（dry-run）。
    if patch --dry-run -p1 -d "$submodule_dir" < "$patch_file" &>/dev/null; then
        echo "  [APPLY] $label"
        patch -p1 -d "$submodule_dir" < "$patch_file"
    else
        echo "  [ALREADY APPLIED or CONFLICT] $label — skipping"
    fi
}

echo "=== Applying adapter patches ==="

apply_patch \
    "third_party/PGM-index" \
    "adapters/pgm_index_patch/pgm_index.patch" \
    "PGM-index PLA injection"

# FITing-Tree patch is informational only (no source changes needed).
echo "  [INFO] FITing-Tree: no source patch needed (reference only)"

apply_patch \
    "third_party/LOFT" \
    "adapters/loft_patch/loft.patch" \
    "LOFT PLA injection"

apply_patch \
    "third_party/Efficient-Disk-Learned-Index" \
    "adapters/pgm_disk_patch/pgm_disk.patch" \
    "PGM-disk PLA injection"

echo "=== Done ==="
