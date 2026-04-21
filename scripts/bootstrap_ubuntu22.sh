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

echo "=== pla-learned-index-bench bootstrap (Ubuntu 22.04) without sudo ==="

# ── 1. 更新 apt 索引 ───────────────────────────────────────────────────────────
apt-get update -qq

# ── 2. 基础编译工具 ────────────────────────────────────────────────────────────
# We will install necessary compilers and tools into a local directory using `apt` or manually.
# For compilers like GCC, we assume they are pre-installed or use a local setup.

# ── 3. Python3 + pip ──────────────────────────────────────────────────────────
# Ensure Python3 and pip are available in the local environment
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Please install Python3 manually."
    exit 1
fi

# Ensure pip is available
if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Installing locally using curl."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
fi

# Install Python dependencies locally with --user
pip3 install --user pyyaml matplotlib numpy

# ── 4. 性能/观测工具 ──────────────────────────────────────────────────────────
# For performance tools, we will install them in the local directory or ask the user to install them manually.

# Example: Install `numactl` and `perf` locally (ensure you have necessary environment)
apt-get install -y numactl linux-tools-common valgrind

# ── 5. jemalloc (LOFT 依赖) ───────────────────────────────────────────────────
# For jemalloc, you can build it locally if it’s not available.
# If it’s already installed, you can skip this step.
wget https://github.com/jemalloc/jemalloc/releases/download/5.2.1/jemalloc-5.2.1.tar.bz2
tar -xjf jemalloc-5.2.1.tar.bz2
cd jemalloc-5.2.1
./configure --prefix=$HOME/.local
make -j$(nproc)
make install

# ── 6. userspace-RCPU (LOFT 依赖) ──────────────────────────────────────────────
# If RCPU library is missing, install it locally.
wget https://github.com/urcu/urcu/releases/download/v0.12.0/urcu-0.12.0.tar.gz
tar -xvf urcu-0.12.0.tar.gz
cd urcu-0.12.0
./configure --prefix=$HOME/.local
make -j$(nproc)
make install

# ── 7. 磁盘工具 ───────────────────────────────────────────────────────────────
# For disk tools, we install them locally or ask users to install via apt.
apt-get install -y fio hdparm util-linux

# ── 8. OpenMP (并行 PLA 构建) ─────────────────────────────────────────────────
# Install OpenMP development tools locally.
apt-get install -y libomp-dev

# ── 9. 可选: Intel oneAPI MKL ─────────────────────────────────────────────────
# Intel oneAPI MKL is optional. If needed, uncomment the section below for installing MKL.
# Uncomment the following lines to install MKL in your local directory
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