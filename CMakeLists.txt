cmake_minimum_required(VERSION 3.22)
project(pla_learned_index_bench CXX)

# ─── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build type default ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ─── PLA algorithm selector ──────────────────────────────────────────────────
# Usage: cmake -DPLA_ALGO=optimal|swing|greedy
set(PLA_ALGO "optimal" CACHE STRING "PLA algorithm: optimal | swing | greedy")
set_property(CACHE PLA_ALGO PROPERTY STRINGS optimal swing greedy)
message(STATUS "PLA_ALGO = ${PLA_ALGO}")
string(TOUPPER "${PLA_ALGO}" PLA_ALGO_UPPER)
add_compile_definitions(PLA_ALGO_${PLA_ALGO_UPPER}=1)

# ─── Optional features ────────────────────────────────────────────────────────
option(USE_MKL      "Link Intel MKL (required by LOFT; set OFF to use fallback)" OFF)
option(USE_JEMALLOC "Link jemalloc"                                              OFF)
option(USE_URCU     "Link userspace-RCU (required by LOFT)"                     OFF)
option(BUILD_TESTS  "Build unit tests"                                           ON)
option(BUILD_ONDISK "Build on-disk benchmarks (needs Efficient-Disk submodule)"  ON)
option(BUILD_DYNAMIC "Build LOFT dynamic benchmark"                              ON)

# ─── Compiler flags ───────────────────────────────────────────────────────────
add_compile_options(
  -Wall -Wextra -Wno-unused-parameter
  $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
  $<$<CONFIG:Debug>:-O0 -g>
)

# ─── Threading ───────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ─── OpenMP (parallel PLA build) ─────────────────────────────────────────────
find_package(OpenMP QUIET)

# ─── jemalloc ────────────────────────────────────────────────────────────────
if(USE_JEMALLOC)
  find_library(JEMALLOC_LIB jemalloc REQUIRED)
  message(STATUS "jemalloc: ${JEMALLOC_LIB}")
endif()

# ─── MKL ─────────────────────────────────────────────────────────────────────
if(USE_MKL)
  find_package(MKL QUIET)
  if(NOT MKL_FOUND)
    message(WARNING "MKL not found; LOFT will be compiled without it (may fail at runtime)")
  endif()
endif()

# ─── URCU ────────────────────────────────────────────────────────────────────
if(USE_URCU)
  find_library(URCU_LIB urcu REQUIRED)
  find_library(URCU_CDS_LIB urcu-cds REQUIRED)
  message(STATUS "urcu: ${URCU_LIB}, urcu-cds: ${URCU_CDS_LIB}")
endif()

# ─── Third-party include paths ───────────────────────────────────────────────
set(PGM_INCLUDE    "${CMAKE_SOURCE_DIR}/third_party/PGM-index/include")
set(FITING_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/FITing-Tree/include")
set(LOFT_INCLUDE   "${CMAKE_SOURCE_DIR}/third_party/LOFT")
set(EDLI_ROOT      "${CMAKE_SOURCE_DIR}/third_party/Efficient-Disk-Learned-Index")
set(SOSD_INCLUDE   "${CMAKE_SOURCE_DIR}/third_party/SOSD")

# ─── PLA library (header-only + thin .cpp) ───────────────────────────────────
add_library(pla_lib INTERFACE)
target_include_directories(pla_lib INTERFACE
  "${CMAKE_SOURCE_DIR}/pla/include"
  "${PGM_INCLUDE}"
  "${FITING_INCLUDE}"
)
target_compile_definitions(pla_lib INTERFACE PLA_ALGO_${PLA_ALGO_UPPER}=1)
if(OpenMP_CXX_FOUND)
  target_link_libraries(pla_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

# ─── Benchmarks ──────────────────────────────────────────────────────────────

# 1. PLA-only build benchmark
add_executable(pla_build_bench bench/pla_only/pla_build_bench.cpp)
target_link_libraries(pla_build_bench PRIVATE pla_lib Threads::Threads)

# 2. In-memory lookup benchmark
add_executable(lookup_bench bench/inmem/lookup_bench.cpp)
target_link_libraries(lookup_bench PRIVATE pla_lib Threads::Threads)
target_include_directories(lookup_bench PRIVATE "${SOSD_INCLUDE}")

# 3. LOFT dynamic benchmark wrapper
if(BUILD_DYNAMIC)
  add_executable(dynamic_bench bench/dynamic/dynamic_bench.cpp)
  target_include_directories(dynamic_bench PRIVATE
    "${LOFT_INCLUDE}"
    "${CMAKE_SOURCE_DIR}/pla/include"
    "${PGM_INCLUDE}"
  )
  target_link_libraries(dynamic_bench PRIVATE pla_lib Threads::Threads)
  if(USE_JEMALLOC)
    target_link_libraries(dynamic_bench PRIVATE ${JEMALLOC_LIB})
  endif()
  if(USE_URCU)
    target_link_libraries(dynamic_bench PRIVATE ${URCU_LIB} ${URCU_CDS_LIB})
  endif()
  if(USE_MKL AND MKL_FOUND)
    target_link_libraries(dynamic_bench PRIVATE MKL::MKL)
  endif()
endif()

# 4. On-disk benchmark wrapper (Efficient-Disk-Learned-Index)
if(BUILD_ONDISK)
  if(EXISTS "${EDLI_ROOT}/CMakeLists.txt")
    # Pull in EDLI's include dirs without recursively adding its full build
    set(EDLI_INCLUDE "${EDLI_ROOT}")
    set(EDLI_PGM_DISK_INCLUDE "${EDLI_ROOT}/indexes/PGM-index-disk")

    add_executable(ondisk_bench bench/ondisk/ondisk_bench.cpp)
    target_include_directories(ondisk_bench PRIVATE
      "${EDLI_INCLUDE}"
      "${EDLI_PGM_DISK_INCLUDE}"
      "${PGM_INCLUDE}"
      "${CMAKE_SOURCE_DIR}/pla/include"
    )
    target_link_libraries(ondisk_bench PRIVATE pla_lib Threads::Threads)
    if(USE_JEMALLOC)
      target_link_libraries(ondisk_bench PRIVATE ${JEMALLOC_LIB})
    endif()
  else()
    message(STATUS "Efficient-Disk-Learned-Index submodule not initialised; skipping ondisk_bench")
  endif()
endif()

# ─── Unit tests ──────────────────────────────────────────────────────────────
if(BUILD_TESTS)
  enable_testing()
  add_executable(pla_unit_test tests/pla_unit_test.cpp)
  target_link_libraries(pla_unit_test PRIVATE pla_lib)
  add_test(NAME pla_unit_test COMMAND pla_unit_test)
endif()

# ─── Install / export (optional) ─────────────────────────────────────────────
message(STATUS "")
message(STATUS "======= pla-learned-index-bench build summary =======")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  PLA_ALGO     : ${PLA_ALGO}")
message(STATUS "  OpenMP       : ${OpenMP_CXX_FOUND}")
message(STATUS "  jemalloc     : ${USE_JEMALLOC}")
message(STATUS "  MKL          : ${USE_MKL}")
message(STATUS "  URCU         : ${USE_URCU}")
message(STATUS "  BUILD_TESTS  : ${BUILD_TESTS}")
message(STATUS "  BUILD_DYNAMIC: ${BUILD_DYNAMIC}")
message(STATUS "  BUILD_ONDISK : ${BUILD_ONDISK}")
message(STATUS "=====================================================")
