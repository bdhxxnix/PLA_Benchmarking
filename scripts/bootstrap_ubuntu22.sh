#!/usr/bin/env bash
# scripts/bootstrap_ubuntu22.sh
# 一键安装 pla-learned-index-bench 所需的所有系统依赖（Ubuntu 22.04）。
# 用法: bash scripts/bootstrap_ubuntu22.sh
#
# 安装内容:
#   - 编译工具: gcc-11, g++-11, clang-14, cmake, ninja-build, make
#   - Python3 + pip + 依赖库 (pyyaml, matplotlib, numpy)
#   - 性能工具: numactl, linux-tools-common (perf), valgrind
#   - 内存分配: libjemalloc-dev
#   - RCPU 库: liburcu-dev (LOFT 依赖)
#   - 磁盘工具: fio, hdparm
#   - 可选 Intel oneAPI MKL (LOFT 完整支持)

set -euo pipefail

echo "=== pla-learned-index-bench bootstrap (without sudo) ==="

# ── 1. 检查并安装conda ─────────────────────────────────────────────────────
# 如果没有conda安装，我们提供 Miniconda 安装。
if ! command -v conda &>/dev/null; then
    echo "Conda not found. Installing Miniconda locally..."
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# 创建一个新的conda环境，并激活它
conda create -y -n pla-env python=3.9
conda activate pla-env

# ── 2. Python3 + pip ──────────────────────────────────────────────────────────
# 安装Python依赖库
pip install --user pyyaml matplotlib numpy

# ── 3. 安装性能/观测工具 ─────────────────────────────────────────────────────
# 使用conda安装性能工具
conda install -y numactl valgrind

# ── 4. jemalloc (LOFT 依赖) ───────────────────────────────────────────────────
# 下载并安装 jemalloc
wget https://github.com/jemalloc/jemalloc/releases/download/5.2.1/jemalloc-5.2.1.tar.bz2
tar -xjf jemalloc-5.2.1.tar.bz2
cd jemalloc-5.2.1
./configure --prefix=$HOME/.local
make -j$(nproc)
make install
cd ..

# ── 5. userspace-RCPU (LOFT 依赖) ──────────────────────────────────────────────
# 下载并安装 RCPU (liburcu)
wget https://github.com/urcu/urcu/releases/download/v0.12.0/urcu-0.12.0.tar.gz
tar -xvf urcu-0.12.0.tar.gz
cd urcu-0.12.0
./configure --prefix=$HOME/.local
make -j$(nproc)
make install
cd ..

# ── 6. 安装磁盘工具 ───────────────────────────────────────────────────────────
# 使用conda安装磁盘工具
conda install -y fio hdparm

# ── 7. OpenMP (并行 PLA 构建) ─────────────────────────────────────────────────
# 使用conda安装OpenMP
conda install -y libomp

# ── 8. 可选: Intel oneAPI MKL ─────────────────────────────────────────────────
# 如果需要 Intel oneAPI MKL，可以解开以下部分
# wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
#     | gpg --dearmor > $HOME/.local/share/keyrings/oneapi-archive-keyring.gpg
# echo "deb [signed-by=$HOME/.local/share/keyrings/oneapi-archive-keyring.gpg] \
#     https://apt.repos.intel.com/oneapi all main" \
#     > $HOME/.local/etc/apt/sources.list.d/oneAPI.list
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