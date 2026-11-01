#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

const char* KERNEL_SRCS = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 2

// -------------------------------------------------------------------------
// Kernel 1: Baseline RPM Wave64 (from bench_rpm_gemm.cpp)
// Single-buffered LDS, 2 barriers per tile, half2 c2[8][4]
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_rpm_baseline(
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

    __local half lds_A[64][32 + LDS_PAD];
    __local half2 lds_B2[32][32 + (LDS_PAD/2)];

    half2 c2[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c2[i][j] = (half2)(0.0h, 0.0h);
    }

    const int load_row_a = tid;
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 1;
    const int load_col_b = (tid & 1) << 5;
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 32;

        half16 a0 = vload16(0, &A[global_row_a * K + k_base + 0]);
        half16 a1 = vload16(0, &A[global_row_a * K + k_base + 16]);
        vstore16(a0, 0, &lds_A[load_row_a][0]);
        vstore16(a1, 0, &lds_A[load_row_a][16]);

        half16 b_raw = vload16(0, &B[(k_base + load_row_b) * N + global_col_b]);
        half16 b_raw2 = vload16(0, &B[(k_base + load_row_b) * N + global_col_b + 16]);

        #pragma unroll
        for (int p = 0; p < 8; p++) {
            lds_B2[load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&b_raw)[p*2], ((half*)&b_raw)[p*2 + 1]);
            lds_B2[load_row_b][(load_col_b >> 1) + 8 + p] = (half2)(((half*)&b_raw2)[p*2], ((half*)&b_raw2)[p*2 + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int my_m = ty * 8;
        const int my_n2 = tx * 4;

        #pragma unroll 16
        for (int k = 0; k < 32; k++) {
            half2 b_pack[4];
            b_pack[0] = lds_B2[k][my_n2 + 0];
            b_pack[1] = lds_B2[k][my_n2 + 1];
            b_pack[2] = lds_B2[k][my_n2 + 2];
            b_pack[3] = lds_B2[k][my_n2 + 3];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                half a_val = lds_A[my_m + i][k];
                half2 a_pack = (half2)(a_val, a_val);

                c2[i][0] = fma(a_pack, b_pack[0], c2[i][0]);
                c2[i][1] = fma(a_pack, b_pack[1], c2[i][1]);
                c2[i][2] = fma(a_pack, b_pack[2], c2[i][2]);
                c2[i][3] = fma(a_pack, b_pack[3], c2[i][3]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < 8; i++) {
        const int out_row = block_m + ty * 8 + i;
        if (out_row < M) {
            __global half* out_ptr = C + out_row * N + block_n + tx * 8;
            half8 out_v = (half8)(c2[i][0].x, c2[i][0].y, c2[i][1].x, c2[i][1].y,
                                 c2[i][2].x, c2[i][2].y, c2[i][3].x, c2[i][3].y);
            vstore8(out_v, 0, out_ptr);
        }
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Double-Buffered LDS with Single Barrier per iteration
// Eliminates 50% of barriers and overlaps DDR4 prefetch with FMA execution!
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_rpm_double_buffered(
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

    // Double buffered ping-pong LDS
    __local half lds_A[2][64][32 + LDS_PAD];
    __local half2 lds_B2[2][32][32 + (LDS_PAD/2)];

    half2 c2[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c2[i][j] = (half2)(0.0h, 0.0h);
    }

    const int load_row_a = tid;
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 1;
    const int load_col_b = (tid & 1) << 5;
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 32;

    // Prologue: Prefetch Tile 0 into registers and store to LDS buffer 0
    half16 reg_a0 = vload16(0, &A[global_row_a * K + 0]);
    half16 reg_a1 = vload16(0, &A[global_row_a * K + 16]);
    vstore16(reg_a0, 0, &lds_A[0][load_row_a][0]);
    vstore16(reg_a1, 0, &lds_A[0][load_row_a][16]);

    half16 reg_b0 = vload16(0, &B[(0 + load_row_b) * N + global_col_b]);
    half16 reg_b1 = vload16(0, &B[(0 + load_row_b) * N + global_col_b + 16]);

    #pragma unroll
    for (int p = 0; p < 8; p++) {
        lds_B2[0][load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&reg_b0)[p*2], ((half*)&reg_b0)[p*2 + 1]);
        lds_B2[0][load_row_b][(load_col_b >> 1) + 8 + p] = (half2)(((half*)&reg_b1)[p*2], ((half*)&reg_b1)[p*2 + 1]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop: Ping-pong double buffering
    for (int t = 0; t < num_tiles; t++) {
        int read_buf = t & 1;
        int write_buf = (t + 1) & 1;

        // Async prefetch next tile (t + 1) from DDR4 into registers while computing tile t
        if (t + 1 < num_tiles) {
            int next_k = (t + 1) * 32;
            reg_a0 = vload16(0, &A[global_row_a * K + next_k + 0]);
            reg_a1 = vload16(0, &A[global_row_a * K + next_k + 16]);
            reg_b0 = vload16(0, &B[(next_k + load_row_b) * N + global_col_b]);
            reg_b1 = vload16(0, &B[(next_k + load_row_b) * N + global_col_b + 16]);
        }

        // Compute Tile t from lds_*[read_buf]
        const int my_m = ty * 8;
        const int my_n2 = tx * 4;

        #pragma unroll 16
        for (int k = 0; k < 32; k++) {
            half2 b_pack[4];
            b_pack[0] = lds_B2[read_buf][k][my_n2 + 0];
            b_pack[1] = lds_B2[read_buf][k][my_n2 + 1];
            b_pack[2] = lds_B2[read_buf][k][my_n2 + 2];
            b_pack[3] = lds_B2[read_buf][k][my_n2 + 3];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                half a_val = lds_A[read_buf][my_m + i][k];
                half2 a_pack = (half2)(a_val, a_val);

                c2[i][0] = fma(a_pack, b_pack[0], c2[i][0]);
                c2[i][1] = fma(a_pack, b_pack[1], c2[i][1]);
                c2[i][2] = fma(a_pack, b_pack[2], c2[i][2]);
                c2[i][3] = fma(a_pack, b_pack[3], c2[i][3]);
            }
        }

        // Store prefetched registers into write_buf for next tile
        if (t + 1 < num_tiles) {
            vstore16(reg_a0, 0, &lds_A[write_buf][load_row_a][0]);
            vstore16(reg_a1, 0, &lds_A[write_buf][load_row_a][16]);

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                lds_B2[write_buf][load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&reg_b0)[p*2], ((half*)&reg_b0)[p*2 + 1]);
                lds_B2[write_buf][load_row_b][(load_col_b >> 1) + 8 + p] = (half2)(((half*)&reg_b1)[p*2], ((half*)&reg_b1)[p*2 + 1]);
            }
        }

        // Exactly ONE barrier per iteration!
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < 8; i++) {
        const int out_row = block_m + ty * 8 + i;
        if (out_row < M) {
            __global half* out_ptr = C + out_row * N + block_n + tx * 8;
            half8 out_v = (half8)(c2[i][0].x, c2[i][0].y, c2[i][1].x, c2[i][1].y,
                                 c2[i][2].x, c2[i][2].y, c2[i][3].x, c2[i][3].y);
            vstore8(out_v, 0, out_ptr);
        }
    }
}

// -------------------------------------------------------------------------
// Kernel 3: High-Occupancy 32-VGPR Micro-Tile (half2 c2[4][4])
// Workgroup: (8, 16) = 128 threads (2 Wave64s). Tile: 64x64 output.
// Thread computes 4 rows x 8 cols = 32 elements. Accumulator VGPRs = 16!
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(8, 16, 1)))
void gemm_rpm_high_occupancy_c4x4(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..15
    const int tid = ty * 8 + tx;    // 0..127

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // Double buffered LDS for 64x32 tile
    __local half lds_A[2][64][32 + LDS_PAD];
    __local half2 lds_B2[2][32][32 + (LDS_PAD/2)];

    // Only 16 accumulator registers (c2[4][4])!
    half2 c2[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c2[i][j] = (half2)(0.0h, 0.0h);
    }

    // 128 threads loading 64x32 elements of A: each thread loads 16 halfs (half8 x 2)
    const int load_row_a = tid >> 1; // 0..63
    const int load_col_a = (tid & 1) << 4; // 0 or 16
    const int global_row_a = block_m + load_row_a;

    // 128 threads loading 32x64 elements of B: each thread loads 16 halfs (half8 x 2)
    const int load_row_b = tid >> 2; // 0..31
    const int load_col_b = (tid & 3) << 4; // 0, 16, 32, 48
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 32;

    // Prologue: Load Tile 0
    half8 a_pre0 = vload8(0, &A[global_row_a * K + 0 + load_col_a]);
    half8 a_pre1 = vload8(0, &A[global_row_a * K + 0 + load_col_a + 8]);
    vstore8(a_pre0, 0, &lds_A[0][load_row_a][load_col_a]);
    vstore8(a_pre1, 0, &lds_A[0][load_row_a][load_col_a + 8]);

    half8 b_pre0 = vload8(0, &B[(0 + load_row_b) * N + global_col_b]);
    half8 b_pre1 = vload8(0, &B[(0 + load_row_b) * N + global_col_b + 8]);
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        lds_B2[0][load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&b_pre0)[p*2], ((half*)&b_pre0)[p*2 + 1]);
        lds_B2[0][load_row_b][(load_col_b >> 1) + 4 + p] = (half2)(((half*)&b_pre1)[p*2], ((half*)&b_pre1)[p*2 + 1]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < num_tiles; t++) {
        int read_buf = t & 1;
        int write_buf = (t + 1) & 1;

        if (t + 1 < num_tiles) {
            int next_k = (t + 1) * 32;
            a_pre0 = vload8(0, &A[global_row_a * K + next_k + load_col_a]);
            a_pre1 = vload8(0, &A[global_row_a * K + next_k + load_col_a + 8]);
            b_pre0 = vload8(0, &B[(next_k + load_row_b) * N + global_col_b]);
            b_pre1 = vload8(0, &B[(next_k + load_row_b) * N + global_col_b + 8]);
        }

        const int my_m = ty * 4; // 16 threads in Y * 4 = 64
        const int my_n2 = tx * 4; // 8 threads in X * 4 half2 = 64 halfs

        #pragma unroll 16
        for (int k = 0; k < 32; k++) {
            half2 b_pack[4];
            b_pack[0] = lds_B2[read_buf][k][my_n2 + 0];
            b_pack[1] = lds_B2[read_buf][k][my_n2 + 1];
            b_pack[2] = lds_B2[read_buf][k][my_n2 + 2];
            b_pack[3] = lds_B2[read_buf][k][my_n2 + 3];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                half a_val = lds_A[read_buf][my_m + i][k];
                half2 a_pack = (half2)(a_val, a_val);

                c2[i][0] = fma(a_pack, b_pack[0], c2[i][0]);
                c2[i][1] = fma(a_pack, b_pack[1], c2[i][1]);
                c2[i][2] = fma(a_pack, b_pack[2], c2[i][2]);
                c2[i][3] = fma(a_pack, b_pack[3], c2[i][3]);
            }
        }

        if (t + 1 < num_tiles) {
            vstore8(a_pre0, 0, &lds_A[write_buf][load_row_a][load_col_a]);
            vstore8(a_pre1, 0, &lds_A[write_buf][load_row_a][load_col_a + 8]);

            #pragma unroll
            for (int p = 0; p < 4; p++) {
                lds_B2[write_buf][load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&b_pre0)[p*2], ((half*)&b_pre0)[p*2 + 1]);
                lds_B2[write_buf][load_row_b][(load_col_b >> 1) + 4 + p] = (half2)(((half*)&b_pre1)[p*2], ((half*)&b_pre1)[p*2 + 1]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < 4; i++) {
        const int out_row = block_m + ty * 4 + i;
        if (out_row < M) {
            __global half* out_ptr = C + out_row * N + block_n + tx * 8;
            half8 out_v = (half8)(c2[i][0].x, c2[i][0].y, c2[i][1].x, c2[i][1].y,
                                 c2[i][2].x, c2[i][2].y, c2[i][3].x, c2[i][3].y);
            vstore8(out_v, 0, out_ptr);
        }
    }
}
)CLC";

void run_bench(cl_context ctx, cl_command_queue queue, cl_kernel k, const char* name, int M, int N, int K, size_t g_ws[2], size_t l_ws[2], cl_mem dA, cl_mem dB, cl_mem dC) {
    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dA);
    clSetKernelArg(k, 4, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 5, sizeof(cl_mem), &dC);

    clEnqueueNDRangeKernel(queue, k, 2, nullptr, g_ws, l_ws, 0, nullptr, nullptr);
    clFinish(queue);

    double best_ms = 1e9;
    for (int i = 0; i < 7; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k, 2, nullptr, g_ws, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        double ms = (double)(t1 - t0) * 1e-6;
        if (ms < best_ms) best_ms = ms;
    }
    double tflops = (2.0 * (double)M * N * K / (best_ms * 1e-3)) * 1e-12;
    std::cout << "  " << std::left << std::setw(38) << name 
              << ": " << std::fixed << std::setprecision(3) << best_ms << " ms | " 
              << std::setprecision(3) << tflops << " TFLOPS\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  GCN 5.0 (Vega 8 gfx90c) Vector Pipeline & LDS Microbenchmarks   \n";
    std::cout << "=================================================================\n";

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, nullptr);

    size_t len = strlen(KERNEL_SRCS);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRCS, &len, nullptr);
    cl_int b_err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (b_err != CL_SUCCESS) {
        size_t lsz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        std::cerr << "Build Log:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel k_base = clCreateKernel(prog, "gemm_rpm_baseline", nullptr);
    cl_kernel k_db = clCreateKernel(prog, "gemm_rpm_double_buffered", nullptr);
    cl_kernel k_c4x4 = clCreateKernel(prog, "gemm_rpm_high_occupancy_c4x4", nullptr);

    int M = 2048, N = 2048, K = 2048;
    size_t bytes = (size_t)M * K * sizeof(uint16_t);
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    size_t g_ws_8x8[2] = { (size_t)N / 8, (size_t)M / 8 };
    size_t l_ws_8x8[2] = { 8, 8 };

    // Notice for c4x4: workgroup is (8, 16). Each thread computes 8 columns x 4 rows.
    // So 8 threads along dim 0 cover 8*8 = 64 cols. Global work size dim 0 = (N / 64) * 8 = N / 8.
    // 16 threads along dim 1 cover 16*4 = 64 rows. Global work size dim 1 = (M / 64) * 16 = M / 4.
    size_t g_ws_8x16[2] = { (size_t)N / 8, (size_t)M / 4 };
    size_t l_ws_8x16[2] = { 8, 16 };

    std::cout << "\n[Experiment 1] Matrix Dimensions: 2048 x 2048 x 2048\n";
    run_bench(ctx, queue, k_base, "Baseline RPM (Single LDS, 2 Bar)", M, N, K, g_ws_8x8, l_ws_8x8, dA, dB, dC);
    run_bench(ctx, queue, k_db, "Double-Buffered LDS (1 Barrier)", M, N, K, g_ws_8x8, l_ws_8x8, dA, dB, dC);
    run_bench(ctx, queue, k_c4x4, "High Occupancy c2[4][4] (32 VGPR)", M, N, K, g_ws_8x16, l_ws_8x16, dA, dB, dC);

    std::cout << "\n[Experiment 2] Large Prefill: 4096 x 2048 x 2048\n";
    int M2 = 4096;
    size_t bytes_A2 = (size_t)M2 * K * sizeof(uint16_t);
    size_t bytes_C2 = (size_t)M2 * N * sizeof(uint16_t);
    cl_mem dA2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes_A2, nullptr, nullptr);
    cl_mem dC2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_C2, nullptr, nullptr);

    size_t g_ws_8x8_2[2] = { (size_t)N / 8, (size_t)M2 / 8 };
    size_t g_ws_8x16_2[2] = { (size_t)N / 8, (size_t)M2 / 4 };

    run_bench(ctx, queue, k_base, "Baseline RPM (Single LDS, 2 Bar)", M2, N, K, g_ws_8x8_2, l_ws_8x8, dA2, dB, dC2);
    run_bench(ctx, queue, k_db, "Double-Buffered LDS (1 Barrier)", M2, N, K, g_ws_8x8_2, l_ws_8x8, dA2, dB, dC2);
    run_bench(ctx, queue, k_c4x4, "High Occupancy c2[4][4] (32 VGPR)", M2, N, K, g_ws_8x16_2, l_ws_8x16, dA2, dB, dC2);

    return 0;
}
