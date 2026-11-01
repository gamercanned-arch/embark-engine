// test_tune.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

const char* kernel_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// Optimized 32x32 Output Tile with 4x4 Micro-tile and Coalesced DDR4 Loads
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_coalesced_32x32_4x4(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63

    const int block_m = get_group_id(1) * 32;
    const int block_n = get_group_id(0) * 32;

    // Transposed LDS for A: [32 cols of K][32 rows of M + PAD]
    // Standard LDS for B:   [32 rows of K][32 cols of N + PAD]
    __local float lds_A_T[32][32 + LDS_PAD];
    __local float lds_B[32][32 + LDS_PAD];

    float4 c[4];
    c[0] = (float4)(0.0f);
    c[1] = (float4)(0.0f);
    c[2] = (float4)(0.0f);
    c[3] = (float4)(0.0f);

    // Coalesced loading mapping:
    // 64 threads loading 32x32 floats (256 float4s):
    // In each of 4 steps, 64 threads load 64 float4s (8 full rows of 32 floats):
    // 8 threads per row (each thread loads 4 floats = 1 float4):
    const int load_row = tid >> 3; // 0..7
    const int load_col = (tid & 7) * 4; // 0, 4, 8, 12, 16, 20, 24, 28

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 32;

        // Load Tile of A and B with 100% coalesced accesses
        #pragma unroll
        for (int s = 0; s < 4; s++)
        {
            int r = load_row + s * 8; // 0..31

            // Load A: row r of M, col load_col of K
            float4 a_val = vload4(0, &A[(block_m + r) * K + k_tile + load_col]);
            // Store transposed into lds_A_T: col of K, row of M
            lds_A_T[load_col + 0][r] = a_val.x;
            lds_A_T[load_col + 1][r] = a_val.y;
            lds_A_T[load_col + 2][r] = a_val.z;
            lds_A_T[load_col + 3][r] = a_val.w;

            // Load B: row r of K, col load_col of N
            float4 b_val = vload4(0, &B[(k_tile + r) * N + block_n + load_col]);
            // Store standard into lds_B
            vstore4(b_val, 0, &lds_B[r][load_col]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 32 outer products using 128-bit vector LDS loads and vector FMAs
        const int row_m = ty * 4;
        const int col_n = tx * 4;

        #pragma unroll 16
        for (int k = 0; k < 32; k++)
        {
            // Vector load 4 floats from lds_A_T: contiguous along M!
            float4 a_vec = vload4(0, &lds_A_T[k][row_m]);
            // Vector load 4 floats from lds_B: contiguous along N!
            float4 b_vec = vload4(0, &lds_B[k][col_n]);

            c[0] = fma((float4)a_vec.x, b_vec, c[0]);
            c[1] = fma((float4)a_vec.y, b_vec, c[1]);
            c[2] = fma((float4)a_vec.z, b_vec, c[2]);
            c[3] = fma((float4)a_vec.w, b_vec, c[3]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write back
    const int out_m = block_m + ty * 4;
    const int out_n = block_n + tx * 4;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if (out_m + i < M) {
            vstore4(c[i], 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// Optimized 64x64 Output Tile with 8x8 Micro-tile and Coalesced DDR4 Loads
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_coalesced_64x64_8x8(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // K_TILE = 16
    // lds_A_T: 16 cols of K x 64 rows of M
    // lds_B:   16 rows of K x 64 cols of N
    __local float lds_A_T[16][64 + LDS_PAD];
    __local float lds_B[16][64 + LDS_PAD];

    float8 c[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) c[i] = (float8)(0.0f);

    // Loading 64x16 of A (1024 floats):
    // 64 threads: each thread loads 16 floats (4 x float4):
    // Row r = tid (0..63), 16 cols loaded as 4 float4s:
    const int load_row_a = tid; // 0..63

    // Loading 16x64 of B (1024 floats):
    // 16 rows of 64 floats. In 1 row, 64 floats = 16 float4s.
    // 64 threads load 4 rows at a time (16 threads per row):
    const int load_row_b = tid >> 4; // 0..3
    const int load_col_b = (tid & 15) * 4; // 0, 4, 8 ... 60

    const int num_tiles = K / 16;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 16;

        // Load A: row load_row_a, 4 float4s along K
        #pragma unroll
        for (int p = 0; p < 4; p++)
        {
            float4 a_val = vload4(0, &A[(block_m + load_row_a) * K + k_tile + p * 4]);
            lds_A_T[p * 4 + 0][load_row_a] = a_val.x;
            lds_A_T[p * 4 + 1][load_row_a] = a_val.y;
            lds_A_T[p * 4 + 2][load_row_a] = a_val.z;
            lds_A_T[p * 4 + 3][load_row_a] = a_val.w;
        }

        // Load B: 16 rows x 64 cols. In 4 passes of 4 rows:
        #pragma unroll
        for (int s = 0; s < 4; s++)
        {
            int r = load_row_b + s * 4;
            float4 b_val = vload4(0, &B[(k_tile + r) * N + block_n + load_col_b]);
            vstore4(b_val, 0, &lds_B[r][load_col_b]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int row_m = ty * 8;
        const int col_n = tx * 8;

        #pragma unroll 16
        for (int k = 0; k < 16; k++)
        {
            float8 a_vec = vload8(0, &lds_A_T[k][row_m]);
            float8 b_vec = vload8(0, &lds_B[k][col_n]);

            c[0] = fma((float8)a_vec.s0, b_vec, c[0]);
            c[1] = fma((float8)a_vec.s1, b_vec, c[1]);
            c[2] = fma((float8)a_vec.s2, b_vec, c[2]);
            c[3] = fma((float8)a_vec.s3, b_vec, c[3]);
            c[4] = fma((float8)a_vec.s4, b_vec, c[4]);
            c[5] = fma((float8)a_vec.s5, b_vec, c[5]);
            c[6] = fma((float8)a_vec.s6, b_vec, c[6]);
            c[7] = fma((float8)a_vec.s7, b_vec, c[7]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int out_m = block_m + ty * 8;
    const int out_n = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        if (out_m + i < M) {
            vstore8(c[i], 0, &C[(out_m + i) * N + out_n]);
        }
    }
}
)CLC";

int main() {
    Vega8Context ctx;
    ctx.init();

    float* h_A = (float*)_aligned_malloc(M * K * sizeof(float), 4096);
    float* h_B = (float*)_aligned_malloc(K * N * sizeof(float), 4096);
    float* h_C = (float*)_aligned_malloc(M * N * sizeof(float), 4096);

    for (int i = 0; i < M * K; ++i) h_A[i] = 0.01f * (i % 13);
    for (int i = 0; i < K * N; ++i) h_B[i] = 0.01f * (i % 17);

    cl_mem d_A = ctx.wrap_host_buffer(h_A, M * K * sizeof(float), CL_MEM_READ_ONLY);
    cl_mem d_B = ctx.wrap_host_buffer(h_B, K * N * sizeof(float), CL_MEM_READ_ONLY);
    cl_mem d_C = ctx.wrap_host_buffer(h_C, M * N * sizeof(float), CL_MEM_READ_WRITE);

    cl_program prog = ctx.build_program_from_source(kernel_src);
    if (!prog) return 1;

    auto test_k = [&](const char* name, size_t g0, size_t g1, size_t l0, size_t l1) {
        cl_kernel k = ctx.create_kernel(prog, name);
        clSetKernelArg(k, 0, sizeof(int), &M);
        clSetKernelArg(k, 1, sizeof(int), &N);
        clSetKernelArg(k, 2, sizeof(int), &K);
        clSetKernelArg(k, 3, sizeof(cl_mem), &d_A);
        clSetKernelArg(k, 4, sizeof(cl_mem), &d_B);
        clSetKernelArg(k, 5, sizeof(cl_mem), &d_C);

        size_t gw[2] = {g0, g1};
        size_t lw[2] = {l0, l1};

        // Warmup
        for (int w = 0; w < 5; ++w) {
            clEnqueueNDRangeKernel(ctx.get_queue(), k, 2, nullptr, gw, lw, 0, nullptr, nullptr);
        }
        ctx.finish();

        double total_ms = 0;
        int iters = 20;
        for (int i = 0; i < iters; ++i) {
            cl_event ev;
            clEnqueueNDRangeKernel(ctx.get_queue(), k, 2, nullptr, gw, lw, 0, nullptr, &ev);
            clWaitForEvents(1, &ev);
            cl_ulong s, e;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr);
            clReleaseEvent(ev);
            total_ms += (e - s) * 1e-6;
        }

        double avg_ms = total_ms / iters;
        double tflops = (TOTAL_FLOPS / (avg_ms * 1e-3)) / 1e12;
        std::cout << "[" << name << "] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;
        clReleaseKernel(k);
    };

    test_k("gemm_coalesced_32x32_4x4", (N/32)*8, (M/32)*8, 8, 8);
    test_k("gemm_coalesced_64x64_8x8", (N/64)*8, (M/64)*8, 8, 8);

    return 0;
}
