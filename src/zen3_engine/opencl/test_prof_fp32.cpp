// test_prof_fp32.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

const char* src_prof_fp32 = R"CLC(
#define LDS_PAD 1

// FP32 Compute Only
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void fp32_compute_only(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    __local float lds_A_T[16][64 + LDS_PAD];
    __local float lds_B[16][64 + LDS_PAD];

    float4 c[8][2];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c[i][0] = (float4)(0.0f);
        c[i][1] = (float4)(0.0f);
    }

    #pragma unroll
    for (int p = 0; p < 16; p++) {
        lds_A_T[p][tid] = 1.0f;
        lds_B[p][tid] = 1.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int row_m = ty * 8;
    const int col_n = tx * 8;

    const int num_tiles = K / 16;

    for (int t = 0; t < num_tiles; t++)
    {
        #pragma unroll 16
        for (int k = 0; k < 16; k++)
        {
            float4 a0 = vload4(0, &lds_A_T[k][row_m + 0]);
            float4 a1 = vload4(0, &lds_A_T[k][row_m + 4]);

            float4 b0 = vload4(0, &lds_B[k][col_n + 0]);
            float4 b1 = vload4(0, &lds_B[k][col_n + 4]);

            c[0][0] += a0.x * b0; c[0][1] += a0.x * b1;
            c[1][0] += a0.y * b0; c[1][1] += a0.y * b1;
            c[2][0] += a0.z * b0; c[2][1] += a0.z * b1;
            c[3][0] += a0.w * b0; c[3][1] += a0.w * b1;

            c[4][0] += a1.x * b0; c[4][1] += a1.x * b1;
            c[5][0] += a1.y * b0; c[5][1] += a1.y * b1;
            c[6][0] += a1.z * b0; c[6][1] += a1.z * b1;
            c[7][0] += a1.w * b0; c[7][1] += a1.w * b1;
        }
    }

    const int out_m = block_m + ty * 8;
    const int out_n = block_n + tx * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (out_m + i < M) {
            vstore4(c[i][0], 0, &C[(out_m + i) * N + out_n + 0]);
            vstore4(c[i][1], 0, &C[(out_m + i) * N + out_n + 4]);
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

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

    cl_mem d_A = ctx.wrap_host_buffer(h_A, M * K * sizeof(float), CL_MEM_READ_ONLY);
    cl_mem d_B = ctx.wrap_host_buffer(h_B, K * N * sizeof(float), CL_MEM_READ_ONLY);
    cl_mem d_C = ctx.wrap_host_buffer(h_C, M * N * sizeof(float), CL_MEM_READ_WRITE);

    cl_program prog = ctx.build_program_from_source(src_prof_fp32);
    if (!prog) return 1;

    cl_kernel k = ctx.create_kernel(prog, "fp32_compute_only");
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
    std::cout << "[fp32_compute_only] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;

    return 0;
}
