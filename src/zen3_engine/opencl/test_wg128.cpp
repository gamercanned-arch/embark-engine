// test_wg128.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

const char* src_wg = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// ============================================================================
// 128 threads (2 Wavefronts): Local (16, 8)
// Tile: M=64, N=128, K_TILE=16.
// Thread micro-tile: 8 rows of M x 8 cols of N (4 half2)
// 8 threads along M * 8 = 64 rows of M.
// 16 threads along N * 8 = 128 cols of N.
// ============================================================================
__kernel __attribute__((reqd_work_group_size(16, 8, 1)))
void gemm_fp16_wg128_64x128(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..15 (N dimension)
    const int ty = get_local_id(1); // 0..7  (M dimension)
    const int tid = ty * 16 + tx;   // 0..127 (128 threads)

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 128;

    // LDS tiles:
    // A: 16 cols of K x 64 rows of M = 1024 halfs = 2 KB
    // B: 16 rows of K x 128 cols of N (64 half2 per row) = 2048 halfs = 4 KB
    __local half lds_A_T[16][64 + LDS_PAD];
    __local half2 lds_B2[16][64 + LDS_PAD];

    half2 c[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            c[i][j] = (half2)(0.0h, 0.0h);
        }
    }

    // 128 threads loading:
    // A: 64 rows x 16 cols = 1024 halfs (64 half16s)
    // 128 threads -> threads 0..63 load 1 row of A (16 halfs = half16)
    const int load_row_a = tid; // 0..127 (only 0..63 used for A)

    // B: 16 rows x 128 cols = 2048 halfs (128 half16s)
    // 128 threads -> each loads exactly 16 halfs (half16 = 8 half2)!
    // row = tid / 8 (0..15), col = (tid % 8) * 8 half2 (0, 8, 16 ... 56)
    const int load_row_b = tid >> 3; // 0..15
    const int load_col_b = (tid & 7) * 8; // 0..56 (in half2)

    const int num_tiles = K / 16;

    for (int t = 0; t < num_tiles; t++)
    {
        const int k_tile = t * 16;

        // Load A: threads 0..63 load 16 halfs
        if (tid < 64) {
            half16 a_raw = vload16(0, &A[(block_m + tid) * K + k_tile]);
            #pragma unroll
            for (int p = 0; p < 16; p++) {
                lds_A_T[p][tid] = ((half*)&a_raw)[p];
            }
        }

        // Load B: all 128 threads load 16 halfs (8 half2)
        half16 b_raw = vload16(0, &B[(k_tile + load_row_b) * N + block_n + load_col_b * 2]);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            lds_B2[load_row_b][load_col_b + p] = (half2)(((half*)&b_raw)[p*2], ((half*)&b_raw)[p*2 + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int row_m = ty * 8;
        const int col_n2 = tx * 4; // 4 half2 = 8 cols

        #pragma unroll 16
        for (int k = 0; k < 16; k++)
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

    cl_program prog = ctx.build_program_from_source(src_wg);
    if (!prog) return 1;

    cl_kernel k = ctx.create_kernel(prog, "gemm_fp16_wg128_64x128");
    if (!k) return 1;

    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(k, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(k, 5, sizeof(cl_mem), &d_C);

    // Global: (N/128)*16, (M/64)*8 = (2048/128)*16 = 256, (512/64)*8 = 64
    size_t gw[2] = {256, 64};
    size_t lw[2] = {16, 8};

    for (int w = 0; w < 5; ++w) {
        cl_int err = clEnqueueNDRangeKernel(ctx.get_queue(), k, 2, nullptr, gw, lw, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Warmup launch failed: " << Vega8Context::get_error_string(err) << std::endl;
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
    std::cout << "[gemm_fp16_wg128_64x128] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;

    return 0;
}
