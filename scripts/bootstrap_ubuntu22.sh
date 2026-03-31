#!/usr/bin/env bash
# scripts/bootstrap_ubuntu22.sh
# 一键安装 pla-learned-index-bench 所需的所有系统依赖（Ubuntu 22.04）。
# 用法: sudo bash scripts/bootstrap_ubuntu22.sh
#
# 安装内容:
#   - 编译工具: gcc-11, g++-11, clang-14, cmake, ninja-build, make
#   - Python3 + pip + 依赖库 (pyyaml, matplotlib, numpy)
#   - 性能工具: numactl, linux-tools-common (perf), valgrind
#   - 内存分配: libjemalloc-dev
#   - RCU 库: liburcu-dev (LOFT 依赖)
#   - 磁盘工具: fio, hdparm
#   - 可选 Intel oneAPI MKL (LOFT 完整支持)
set -euo pipefail

echo "=== pla-learned-index-bench bootstrap (Ubuntu 22.04) ==="

# ── 1. 更新 apt 索引 ───────────────────────────────────────────────────────────
apt-get update -qq

# ── 2. 基础编译工具 ────────────────────────────────────────────────────────────
apt-get install -y \
    build-essential \
    gcc-11 g++-11 \
    clang-14 \
    cmake \
    ninja-build \
    git \
    wget curl \
    pkg-config

# 将 gcc-11/g++-11 设为默认（如果系统还没有更新版本）
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110

# ── 3. Python3 + pip ──────────────────────────────────────────────────────────
apt-get install -y python3 python3-pip python3-dev

pip3 install --quiet pyyaml matplotlib numpy

# ── 4. 性能/观测工具 ──────────────────────────────────────────────────────────
KERNEL_VER=$(uname -r)
apt-get install -y \
    numactl \
    linux-tools-common \
    "linux-tools-${KERNEL_VER}" || \
    apt-get install -y linux-tools-generic || true

apt-get install -y valgrind

# ── 5. jemalloc (LOFT 依赖) ───────────────────────────────────────────────────
apt-get install -y libjemalloc-dev

# ── 6. userspace-RCU (LOFT 依赖) ──────────────────────────────────────────────
apt-get install -y liburcu-dev

# ── 7. 磁盘工具 ───────────────────────────────────────────────────────────────
apt-get install -y fio hdparm util-linux

# ── 8. OpenMP (并行 PLA 构建) ─────────────────────────────────────────────────
apt-get install -y libomp-dev

# ── 9. 可选: Intel oneAPI MKL ─────────────────────────────────────────────────
# 取消注释以安装 MKL（需要 ~2GB 空间）。
# LOFT 完整运行需要 MKL；不装时 LOFT 仅能编译不能运行。
#
# wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
#     | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg
# echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] \
#     https://apt.repos.intel.com/oneapi all main" \
#     > /etc/apt/sources.list.d/oneAPI.list
# apt-get update -qq
# apt-get install -y intel-oneapi-mkl-devel

echo ""
echo "=== Bootstrap complete ==="
echo "gcc:     $(gcc --version | head -1)"
echo "cmake:   $(cmake --version | head -1)"
echo "python3: $(python3 --version)"
echo "perf:    $(perf --version 2>&1 | head -1 || echo 'not found (check kernel version)')"
echo ""
echo "Next steps:"
echo "  git clone --recursive <this repo>"
echo "  bash scripts/apply_patches.sh"
echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build -j\$(nproc)"
echo "  python3 tools/runner/run.py --config configs/exp_example.yaml --smoke"
