// test_32x32_half2.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

const char* src_32x32 = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_half2_32x32_4x4(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    const int block_m = get_group_id(1) * 32;
    const int block_n = get_group_id(0) * 32;

    // LDS: 32x32 halfs = 16 half2 per row
    // Stride 17 half2 prevents bank conflicts
    __local half lds_A_T[32][32 + LDS_PAD];
    __local half2 lds_B2[32][16 + LDS_PAD];

    // 4 rows x 2 half2 (4 cols) = 8 VGPRs!
    half2 c[4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[i][0] = (half2)(0.0h, 0.0h);
        c[i][1] = (half2)(0.0h, 0.0h);
    }

    // 64 threads loading 32x32 elements (1024 halfs = 512 half2)
    // Each thread loads 16 halfs (1 x half16) of A and B
    // A: 32 rows x 32 cols -> 2 threads per row (each loads 16 halfs = half16)
    const int load_row_a = tid >> 1; // 0..31
    const int load_col_a = (tid & 1) << 4; // 0 or 16

    // B: 32 rows x 32 cols -> 2 threads per row (each loads 16 halfs = 8 half2)
    const int load_row_b = tid >> 1; // 0..31
    const int load_col_b2 = (tid & 1) << 3; // 0 or 8 half2

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 32;

        // Coalesced DDR4 vector load of A
        half16 a_raw = vload16(0, &A[(block_m + load_row_a) * K + k_tile + load_col_a]);
        #pragma unroll
        for (int p = 0; p < 16; p++) {
            lds_A_T[load_col_a + p][load_row_a] = ((half*)&a_raw)[p];
        }

        // Coalesced DDR4 vector load of B
        half16 b_raw = vload16(0, &B[(k_tile + load_row_b) * N + block_n + load_col_b2 * 2]);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            lds_B2[load_row_b][load_col_b2 + p] = (half2)(((half*)&b_raw)[p*2], ((half*)&b_raw)[p*2 + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int row_m = ty * 4;
        const int col_n2 = tx * 2; // 2 half2 = 4 cols

        #pragma unroll 32
        for (int k = 0; k < 32; k++)
        {
            half a0 = lds_A_T[k][row_m + 0];
            half a1 = lds_A_T[k][row_m + 1];
            half a2 = lds_A_T[k][row_m + 2];
            half a3 = lds_A_T[k][row_m + 3];

            half2 b0 = lds_B2[k][col_n2 + 0];
            half2 b1 = lds_B2[k][col_n2 + 1];

            c[0][0] = fma((half2)(a0, a0), b0, c[0][0]);
            c[0][1] = fma((half2)(a0, a0), b1, c[0][1]);

            c[1][0] = fma((half2)(a1, a1), b0, c[1][0]);
            c[1][1] = fma((half2)(a1, a1), b1, c[1][1]);

            c[2][0] = fma((half2)(a2, a2), b0, c[2][0]);
            c[2][1] = fma((half2)(a2, a2), b1, c[2][1]);

            c[3][0] = fma((half2)(a3, a3), b0, c[3][0]);
            c[3][1] = fma((half2)(a3, a3), b1, c[3][1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int out_m = block_m + ty * 4;
    const int out_n = block_n + tx * 4;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        if (out_m + i < M) {
            half4 out_val = (half4)(
                c[i][0].x, c[i][0].y,
                c[i][1].x, c[i][1].y
            );
            vstore4(out_val, 0, &C[(out_m + i) * N + out_n]);
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

    cl_program prog = ctx.build_program_from_source(src_32x32);
    if (!prog) return 1;

    cl_kernel k = ctx.create_kernel(prog, "gemm_half2_32x32_4x4");
    if (!k) return 1;

    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(k, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(k, 5, sizeof(cl_mem), &d_C);

    size_t gw[2] = {(size_t)(N/32)*8, (size_t)(M/32)*8};
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
    std::cout << "[gemm_half2_32x32_4x4] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;

    return 0;
}
