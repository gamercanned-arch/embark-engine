// test_pingpong16.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

const char* pingpong_src = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 1

// K_TILE = 16 -> LDS per buffer is 16 * 65 * 4 = 4160 bytes.
// Total LDS for 2 buffers: 2 * 4160 * 2 = 16640 bytes (well under 32768 limit).
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_pingpong_fp32_64x64_k16(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // Ping-Pong Double-Buffered LDS
    __local float lds_A[2][16][64 + LDS_PAD];
    __local float lds_B[2][16][64 + LDS_PAD];

    float c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            c[i][j] = 0.0f;
        }
    }

    // 64 threads loading 64x16 elements (1024 floats) = 16 floats (4 float4s) per thread
    // Loading A: thread tid loads 1 row of M (0..63), 16 cols along K (4 float4s)
    const int load_row_a = tid; // 0..63

    // Loading B: 16 rows x 64 cols. 1 row = 16 float4s.
    // 64 threads load 4 rows at a time (16 threads per row):
    const int load_row_b = tid >> 4; // 0..3
    const int load_col_b = (tid & 15) * 4; // 0, 4, 8 ... 60

    // Load Tile 0 into LDS Buffer 0
    #pragma unroll
    for (int p = 0; p < 4; p++)
    {
        float4 a_val = vload4(0, &A[(block_m + load_row_a) * K + p * 4]);
        lds_A[0][p * 4 + 0][load_row_a] = a_val.x;
        lds_A[0][p * 4 + 1][load_row_a] = a_val.y;
        lds_A[0][p * 4 + 2][load_row_a] = a_val.z;
        lds_A[0][p * 4 + 3][load_row_a] = a_val.w;
    }

    #pragma unroll
    for (int s = 0; s < 4; s++)
    {
        int r = load_row_b + s * 4; // 0..15
        float4 b_val = vload4(0, &B[r * N + block_n + load_col_b]);
        vstore4(b_val, 0, &lds_B[0][r][load_col_b]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int num_tiles = K / 16;
    const int row_m = ty * 8;
    const int col_n = tx * 8;

    for (int t = 0; t < num_tiles; t++)
    {
        const int curr_buf = t & 1;
        const int next_buf = (t + 1) & 1;
        const int next_k = (t + 1) * 16;

        float4 pre_a[4];
        float4 pre_b[4];

        if (t + 1 < num_tiles)
        {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                pre_a[p] = vload4(0, &A[(block_m + load_row_a) * K + next_k + p * 4]);
            }

            #pragma unroll
            for (int s = 0; s < 4; s++) {
                int r = load_row_b + s * 4;
                pre_b[s] = vload4(0, &B[(next_k + r) * N + block_n + load_col_b]);
            }
        }

        #pragma unroll 16
        for (int k = 0; k < 16; k++)
        {
            float8 a_vec = vload8(0, &lds_A[curr_buf][k][row_m]);
            float8 b_vec = vload8(0, &lds_B[curr_buf][k][col_n]);

            c[0][0] += a_vec.s0 * b_vec.s0; c[0][1] += a_vec.s0 * b_vec.s1; c[0][2] += a_vec.s0 * b_vec.s2; c[0][3] += a_vec.s0 * b_vec.s3;
            c[0][4] += a_vec.s0 * b_vec.s4; c[0][5] += a_vec.s0 * b_vec.s5; c[0][6] += a_vec.s0 * b_vec.s6; c[0][7] += a_vec.s0 * b_vec.s7;

            c[1][0] += a_vec.s1 * b_vec.s0; c[1][1] += a_vec.s1 * b_vec.s1; c[1][2] += a_vec.s1 * b_vec.s2; c[1][3] += a_vec.s1 * b_vec.s3;
            c[1][4] += a_vec.s1 * b_vec.s4; c[1][5] += a_vec.s1 * b_vec.s5; c[1][6] += a_vec.s1 * b_vec.s6; c[1][7] += a_vec.s1 * b_vec.s7;

            c[2][0] += a_vec.s2 * b_vec.s0; c[2][1] += a_vec.s2 * b_vec.s1; c[2][2] += a_vec.s2 * b_vec.s2; c[2][3] += a_vec.s2 * b_vec.s3;
            c[2][4] += a_vec.s2 * b_vec.s4; c[2][5] += a_vec.s2 * b_vec.s5; c[2][6] += a_vec.s2 * b_vec.s6; c[2][7] += a_vec.s2 * b_vec.s7;

            c[3][0] += a_vec.s3 * b_vec.s0; c[3][1] += a_vec.s3 * b_vec.s1; c[3][2] += a_vec.s3 * b_vec.s2; c[3][3] += a_vec.s3 * b_vec.s3;
            c[3][4] += a_vec.s3 * b_vec.s4; c[3][5] += a_vec.s3 * b_vec.s5; c[3][6] += a_vec.s3 * b_vec.s6; c[3][7] += a_vec.s3 * b_vec.s7;

            c[4][0] += a_vec.s4 * b_vec.s0; c[4][1] += a_vec.s4 * b_vec.s1; c[4][2] += a_vec.s4 * b_vec.s2; c[4][3] += a_vec.s4 * b_vec.s3;
            c[4][4] += a_vec.s4 * b_vec.s4; c[4][5] += a_vec.s4 * b_vec.s5; c[4][6] += a_vec.s4 * b_vec.s6; c[4][7] += a_vec.s4 * b_vec.s7;

            c[5][0] += a_vec.s5 * b_vec.s0; c[5][1] += a_vec.s5 * b_vec.s1; c[5][2] += a_vec.s5 * b_vec.s2; c[5][3] += a_vec.s5 * b_vec.s3;
            c[5][4] += a_vec.s5 * b_vec.s4; c[5][5] += a_vec.s5 * b_vec.s5; c[5][6] += a_vec.s5 * b_vec.s6; c[5][7] += a_vec.s5 * b_vec.s7;

            c[6][0] += a_vec.s6 * b_vec.s0; c[6][1] += a_vec.s6 * b_vec.s1; c[6][2] += a_vec.s6 * b_vec.s2; c[6][3] += a_vec.s6 * b_vec.s3;
            c[6][4] += a_vec.s6 * b_vec.s4; c[6][5] += a_vec.s6 * b_vec.s5; c[6][6] += a_vec.s6 * b_vec.s6; c[6][7] += a_vec.s6 * b_vec.s7;

            c[7][0] += a_vec.s7 * b_vec.s0; c[7][1] += a_vec.s7 * b_vec.s1; c[7][2] += a_vec.s7 * b_vec.s2; c[7][3] += a_vec.s7 * b_vec.s3;
            c[7][4] += a_vec.s7 * b_vec.s4; c[7][5] += a_vec.s7 * b_vec.s5; c[7][6] += a_vec.s7 * b_vec.s6; c[7][7] += a_vec.s7 * b_vec.s7;
        }

        if (t + 1 < num_tiles)
        {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                lds_A[next_buf][p * 4 + 0][load_row_a] = pre_a[p].x;
                lds_A[next_buf][p * 4 + 1][load_row_a] = pre_a[p].y;
                lds_A[next_buf][p * 4 + 2][load_row_a] = pre_a[p].z;
                lds_A[next_buf][p * 4 + 3][load_row_a] = pre_a[p].w;
            }

            #pragma unroll
            for (int s = 0; s < 4; s++) {
                int r = load_row_b + s * 4;
                vstore4(pre_b[s], 0, &lds_B[next_buf][r][load_col_b]);
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
            float8 out_val = (float8)(
                c[i][0], c[i][1], c[i][2], c[i][3],
                c[i][4], c[i][5], c[i][6], c[i][7]
            );
            vstore8(out_val, 0, &C[(out_m + i) * N + out_n]);
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

    cl_program prog = ctx.build_program_from_source(pingpong_src);
    if (!prog) return 1;

    cl_kernel k = ctx.create_kernel(prog, "gemm_pingpong_fp32_64x64_k16");
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
            std::cerr << "Warmup launch failed: " << Vega8Context::get_error_string(err) << std::endl;
            return 1;
        }
    }
    ctx.finish();

    double total_ms = 0;
    int iters = 25;
    for (int i = 0; i < iters; ++i) {
        cl_event ev;
        cl_int err = clEnqueueNDRangeKernel(ctx.get_queue(), k, 2, nullptr, gw, lw, 0, nullptr, &ev);
        if (err != CL_SUCCESS) {
            std::cerr << "Bench launch failed: " << Vega8Context::get_error_string(err) << std::endl;
            return 1;
        }
        clWaitForEvents(1, &ev);
        cl_ulong s, e;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr);
        clReleaseEvent(ev);
        total_ms += (e - s) * 1e-6;
    }

    double avg_ms = total_ms / iters;
    double tflops = (TOTAL_FLOPS / (avg_ms * 1e-3)) / 1e12;
    std::cout << "[gemm_pingpong_fp32_64x64_k16] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;

    return 0;
}
