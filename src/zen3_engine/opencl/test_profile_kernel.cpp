// test_profile_kernel.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

const char* prof_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define LDS_PAD 1

// Case 1: Compute Only (A and B loaded once into LDS outside loop, inner loop computes 64 tiles)
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void kernel_compute_only(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * 8 + tx;

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    __local half lds_A_T[32][64 + LDS_PAD];
    __local half2 lds_B2[32][32 + LDS_PAD];

    half2 c[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = (half2)(0.0h, 0.0h);
    }

    // Init LDS
    #pragma unroll
    for (int p = 0; p < 32; p++) {
        lds_A_T[p][tid] = (half)1.0h;
        lds_B2[p][tid >> 1] = (half2)(1.0h, 1.0h);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int row_m = ty * 8;
    const int col_n2 = tx * 4;

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++)
    {
        #pragma unroll 32
        for (int k = 0; k < 32; k++)
        {
            half a[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) a[i] = lds_A_T[k][row_m + i];

            half2 b[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) b[j] = lds_B2[k][col_n2 + j];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                half2 a_rep = (half2)(a[i], a[i]);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    c[i][j] = fma(a_rep, b[j], c[i][j]);
                }
            }
        }
    }

    const int out_m = block_m + ty * 8;
    const int out_n = block_n + tx * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (out_m + i < M) {
            half8 out_val = (half8)(c[i][0].x, c[i][0].y, c[i][1].x, c[i][1].y, c[i][2].x, c[i][2].y, c[i][3].x, c[i][3].y);
            vstore8(out_val, 0, &C[(out_m + i) * N + out_n]);
        }
    }
}

// Case 2: Memory Load Only (Loads all tiles from global into registers, no compute)
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void kernel_load_only(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * 8 + tx;

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    const int load_row_a = tid;
    const int load_row_b = tid >> 2;
    const int load_col_b = (tid & 3) * 8;

    half16 sum_a = (half16)(0.0h);
    half16 sum_b = (half16)(0.0h);

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 32;
        half16 a0 = vload16(0, &A[(block_m + load_row_a) * K + k_tile + 0]);
        half16 a1 = vload16(0, &A[(block_m + load_row_a) * K + k_tile + 16]);
        sum_a += a0 + a1;

        #pragma unroll
        for (int s = 0; s < 2; s++) {
            int r = load_row_b + s * 16;
            half16 b_raw = vload16(0, &B[(k_tile + r) * N + block_n + load_col_b * 2]);
            sum_b += b_raw;
        }
    }

    C[tid] = sum_a.s0 + sum_b.s0;
}
)CLC";

int main() {
    Vega8Context ctx;
    ctx.init();

    uint16_t* h_A = (uint16_t*)_aligned_malloc(M * K * sizeof(uint16_t), 4096);
    uint16_t* h_B = (uint16_t*)_aligned_malloc(K * N * sizeof(uint16_t), 4096);
    uint16_t* h_C = (uint16_t*)_aligned_malloc(M * N * sizeof(uint16_t), 4096);

    for (int i = 0; i < M * K; ++i) h_A[i] = 0x3C00;
    for (int i = 0; i < K * N; ++i) h_B[i] = 0x3C00;

    cl_mem d_A = ctx.wrap_host_buffer(h_A, M * K * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_B = ctx.wrap_host_buffer(h_B, K * N * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_C = ctx.wrap_host_buffer(h_C, M * N * sizeof(uint16_t), CL_MEM_READ_WRITE);

    cl_program prog = ctx.build_program_from_source(prof_src);
    if (!prog) return 1;

    auto test_fn = [&](const char* name) {
        cl_kernel k = ctx.create_kernel(prog, name);
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
        std::cout << "[" << name << "] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;
        clReleaseKernel(k);
    };

    test_fn("kernel_compute_only");
    test_fn("kernel_load_only");

    return 0;
}
