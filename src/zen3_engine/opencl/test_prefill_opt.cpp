// test_prefill_opt.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

const char* opt_kernels_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// ============================================================================
// Kernel A: Wave64 (64 threads) FP16 GEMM with Double-Buffered Prefetch
// Tile: 64x64, K_TILE = 32, Micro-Tile: 8x8 (32 half2 VGPRs)
// ============================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_wave64_fp16_db(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // LDS double-buffering or single LDS with VGPR prefetch
    // Stride 65 prevents 32-bank conflict on half2 (65 half2 = 130 halfs)
    __local half2 lds_A2[32][32 + LDS_PAD]; // 32 rows of K x 32 half2 (64 halfs of M)
    __local half2 lds_B2[32][32 + LDS_PAD]; // 32 rows of K x 32 half2 (64 halfs of N)

    // 8x8 micro-tile per thread: 8 rows of M x 4 half2 (8 cols of N)
    half2 c[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            c[i][j] = (half2)(0.0h, 0.0h);
        }
    }

    // Coalesced loading coordinates:
    // 64 threads loading 64x32 elements (2048 halfs = 1024 half2)
    // 64 threads -> each thread loads 16 half2 (32 halfs):
    // For A: 32 rows of K, 32 half2 of M.
    // In each step, 64 threads load 2 full rows of 32 half2 (64 half2 total):
    // tid 0..31 load row 0 (32 half2). tid 32..63 load row 1 (32 half2).
    const int a_row_base = (tid >> 5); // 0 or 1
    const int a_col = (tid & 31);     // 0..31 half2

    // For B: 32 rows of K, 32 half2 of N.
    const int b_row_base = (tid >> 5); // 0 or 1
    const int b_col = (tid & 31);     // 0..31 half2

    // VGPR prefetch buffers
    half2 pre_a[16];
    half2 pre_b[16];

    const int num_tiles = K / 32;

    // Initial load: Tile 0 into prefetch VGPRs
    #pragma unroll
    for (int s = 0; s < 16; s++) {
        int r = a_row_base + s * 2;
        // Global A is [M, K]. Transposed in LDS as [K, M]:
        // In A: row is (block_m + a_col*2), col is (r).
        // To keep global loads 100% coalesced:
        // Load from A: row = block_m + (tid % 64), cols = k_tile + (tid / 64)
        // Let's use direct half16 loads:
        int g_row_a = block_m + tid; // row of M (0..63)
        // 32 halfs along K = two half16:
        // prefetch into pre_a
    }

    // Let's use the ultra-clean mapping:
    // Thread tid loads row tid (0..63) of A: 32 halfs along K = two half16!
    // Thread tid loads B: 16 rows of 4 half2:
    const int load_row_b = tid >> 2; // 0..15
    const int load_col_b = (tid & 3) * 8; // 0, 8, 16, 24 half2

    // Preload Tile 0 into VGPRs
    half16 a_raw0 = vload16(0, &A[(block_m + tid) * K + 0]);
    half16 a_raw1 = vload16(0, &A[(block_m + tid) * K + 16]);

    half16 b_raw0 = vload16(0, &B[(0 + load_row_b) * N + block_n + load_col_b * 2]);
    half16 b_raw1 = vload16(0, &B[(16 + load_row_b) * N + block_n + load_col_b * 2]);

    // Store tile 0 into LDS
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        // A transposed: lds_A2[k][m_half2]
        half2 val0 = (half2)(((half*)&a_raw0)[p*2], ((half*)&a_raw0)[p*2 + 1]);
        half2 val1 = (half2)(((half*)&a_raw1)[p*2], ((half*)&a_raw1)[p*2 + 1]);
        lds_A2[p * 2 + 0][tid >> 1] = (tid & 1) ? (half2)(lds_A2[p * 2 + 0][tid >> 1].x, val0.x) : (half2)(val0.x, val0.y);
    }
}
)CLC";

int main() {
    std::cout << "Testing OpenCL optimization options..." << std::endl;
    return 0;
}
