// test_fp16.cpp
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

const char* fp16_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// FP16 with Rapid Packed Math: half2
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_fp16_rpm_64x64_8x8(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // K_TILE = 32
    // A: 32 cols of K x 64 rows of M
    // B: 32 rows of K x 64 cols of N
    __local half lds_A_T[32][64 + LDS_PAD];
    __local half lds_B[32][64 + LDS_PAD];

    half8 c[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) c[i] = (half8)(0.0h);

    // 64 threads loading 64x32 of A = 2048 halfs = 128 half16s
    // Each thread loads 32 halfs (2 x half16)
    const int load_row_a = tid; // 0..63

    // Loading 32x64 of B = 2048 halfs
    // 32 rows of 64 halfs. In 1 row, 64 halfs = 4 half16s.
    // 64 threads load 16 rows at a time (4 threads per row):
    const int load_row_b = tid >> 2; // 0..15
    const int load_col_b = (tid & 3) * 16; // 0, 16, 32, 48

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 32;

        // Load A: row load_row_a, 32 cols along K
        half16 a0 = vload16(0, &A[(block_m + load_row_a) * K + k_tile + 0]);
        half16 a1 = vload16(0, &A[(block_m + load_row_a) * K + k_tile + 16]);

        #pragma unroll
        for (int p = 0; p < 16; p++) {
            lds_A_T[p][load_row_a] = ((half*)&a0)[p];
            lds_A_T[16 + p][load_row_a] = ((half*)&a1)[p];
        }

        // Load B: 32 rows x 64 cols in 2 passes
        #pragma unroll
        for (int s = 0; s < 2; s++)
        {
            int r = load_row_b + s * 16;
            half16 b_val = vload16(0, &B[(k_tile + r) * N + block_n + load_col_b]);
            vstore16(b_val, 0, &lds_B[r][load_col_b]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int row_m = ty * 8;
        const int col_n = tx * 8;

        #pragma unroll 32
        for (int k = 0; k < 32; k++)
        {
            half8 a_vec = vload8(0, &lds_A_T[k][row_m]);
            half8 b_vec = vload8(0, &lds_B[k][col_n]);

            c[0] = fma((half8)a_vec.s0, b_vec, c[0]);
            c[1] = fma((half8)a_vec.s1, b_vec, c[1]);
            c[2] = fma((half8)a_vec.s2, b_vec, c[2]);
            c[3] = fma((half8)a_vec.s3, b_vec, c[3]);
            c[4] = fma((half8)a_vec.s4, b_vec, c[4]);
            c[5] = fma((half8)a_vec.s5, b_vec, c[5]);
            c[6] = fma((half8)a_vec.s6, b_vec, c[6]);
            c[7] = fma((half8)a_vec.s7, b_vec, c[7]);
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

    uint16_t* h_A = (uint16_t*)_aligned_malloc(M * K * sizeof(uint16_t), 4096);
    uint16_t* h_B = (uint16_t*)_aligned_malloc(K * N * sizeof(uint16_t), 4096);
    uint16_t* h_C = (uint16_t*)_aligned_malloc(M * N * sizeof(uint16_t), 4096);

    for (int i = 0; i < M * K; ++i) h_A[i] = 0x3C00; // 1.0h
    for (int i = 0; i < K * N; ++i) h_B[i] = 0x3C00; // 1.0h

    cl_mem d_A = ctx.wrap_host_buffer(h_A, M * K * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_B = ctx.wrap_host_buffer(h_B, K * N * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_C = ctx.wrap_host_buffer(h_C, M * N * sizeof(uint16_t), CL_MEM_READ_WRITE);

    cl_program prog = ctx.build_program_from_source(fp16_src);
    if (!prog) return 1;

    cl_kernel k = ctx.create_kernel(prog, "gemm_fp16_rpm_64x64_8x8");
    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(k, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(k, 5, sizeof(cl_mem), &d_C);

    size_t gw[2] = {(size_t)(N/64)*8, (size_t)(M/64)*8};
    size_t lw[2] = {8, 8};

    for (int w = 0; w < 5; ++w) {
        clEnqueueNDRangeKernel(ctx.get_queue(), k, 2, nullptr, gw, lw, 0, nullptr, nullptr);
    }
    ctx.finish();

    double total_ms = 0;
    int iters = 25;
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
    std::cout << "[gemm_fp16_rpm_64x64_8x8] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;

    return 0;
}
