// test_k64.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

const char* k64_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_half2_k64(
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

    // K_TILE = 64
    // A: 64 cols of K x 64 rows of M = 4096 halfs = 8.2 KB
    // B: 64 rows of K x 32 half2 of N = 4096 halfs = 8.2 KB
    // Total LDS = 16.4 KB
    __local half lds_A_T[64][64 + LDS_PAD];
    __local half2 lds_B2[64][32 + LDS_PAD];

    half2 c[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            c[i][j] = (half2)(0.0h, 0.0h);
        }
    }

    const int load_row_a = tid; // 0..63
    const int load_row_b = tid >> 1; // 0..31
    const int load_col_b = (tid & 1) * 16; // 0 or 16 half2 (32 halfs)

    const int num_tiles = K / 64;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 64;

        // Load 64 halfs of A for row load_row_a
        half16 a0 = vload16(0, &A[(block_m + load_row_a) * K + k_tile + 0]);
        half16 a1 = vload16(0, &A[(block_m + load_row_a) * K + k_tile + 16]);
        half16 a2 = vload16(0, &A[(block_m + load_row_a) * K + k_tile + 32]);
        half16 a3 = vload16(0, &A[(block_m + load_row_a) * K + k_tile + 48]);

        #pragma unroll
        for (int p = 0; p < 16; p++) {
            lds_A_T[0  + p][load_row_a] = ((half*)&a0)[p];
            lds_A_T[16 + p][load_row_a] = ((half*)&a1)[p];
            lds_A_T[32 + p][load_row_a] = ((half*)&a2)[p];
            lds_A_T[48 + p][load_row_a] = ((half*)&a3)[p];
        }

        // Load 64 rows x 64 cols of B:
        // 64 threads load 32 rows (32 halfs each) in 2 passes
        #pragma unroll
        for (int s = 0; s < 2; s++)
        {
            int r = load_row_b + s * 32;
            half16 b0 = vload16(0, &B[(k_tile + r) * N + block_n + load_col_b * 2 + 0]);
            half16 b1 = vload16(0, &B[(k_tile + r) * N + block_n + load_col_b * 2 + 16]);

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                lds_B2[r][load_col_b + 0 + p] = (half2)(((half*)&b0)[p*2], ((half*)&b0)[p*2 + 1]);
                lds_B2[r][load_col_b + 8 + p] = (half2)(((half*)&b1)[p*2], ((half*)&b1)[p*2 + 1]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int row_m = ty * 8;
        const int col_n2 = tx * 4;

        #pragma unroll 64
        for (int k = 0; k < 64; k++)
        {
            half a[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a[i] = lds_A_T[k][row_m + i];
            }

            half2 b[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                b[j] = lds_B2[k][col_n2 + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                half2 a_rep = (half2)(a[i], a[i]);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    c[i][j] = fma(a_rep, b[j], c[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int out_m = block_m + ty * 8;
    const int out_n = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        if (out_m + i < M) {
            half8 out_val = (half8)(
                c[i][0].x, c[i][0].y,
                c[i][1].x, c[i][1].y,
                c[i][2].x, c[i][2].y,
                c[i][3].x, c[i][3].y
            );
            vstore8(out_val, 0, &C[(out_m + i) * N + out_n]);
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

    for (int i = 0; i < M * K; ++i) h_A[i] = 0x3C00;
    for (int i = 0; i < K * N; ++i) h_B[i] = 0x3C00;

    cl_mem d_A = ctx.wrap_host_buffer(h_A, M * K * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_B = ctx.wrap_host_buffer(h_B, K * N * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_C = ctx.wrap_host_buffer(h_C, M * N * sizeof(uint16_t), CL_MEM_READ_WRITE);

    cl_program prog = ctx.build_program_from_source(k64_src);
    if (!prog) return 1;

    cl_kernel k = ctx.create_kernel(prog, "gemm_half2_k64");
    if (!k) return 1;

    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(k, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(k, 5, sizeof(cl_mem), &d_C);

    size_t gw[2] = {(size_t)(N/64)*8, (size_t)(M/64)*8};
    size_t lw[2] = {8, 8};

    for (int w = 0; w < 5; ++w) {
        cl_int err = clEnqueueNDRangeKernel(ctx.get_queue(), k, 2, nullptr, gw, lw, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Warmup failed: " << Vega8Context::get_error_string(err) << std::endl;
            return 1;
        }
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
    std::cout << "[gemm_half2_k64] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;

    return 0;
}
