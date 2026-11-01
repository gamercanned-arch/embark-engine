#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>

const char* KERNEL_SRCS_2 = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 2

// -------------------------------------------------------------------------
// Kernel: 128x128 Tile with 256 threads (4 Wave64s per Workgroup)
// Each thread computes 8x8 micro-tile (c2[8][4]) in packed half2.
// K-step = 16.
// Total LDS = 128 * 18 * 2 + 16 * 130 * 2 = 4608 + 4160 = 8.7 KB!
// With 8.7 KB LDS and 4 waves per workgroup:
// A CU can host up to 6 workgroups = 24 active wavefronts!
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_rpm_128x128_w256(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..15
    const int ty = get_local_id(1); // 0..15
    const int tid = ty * 16 + tx;   // 0..255

    const int block_m = get_group_id(1) * 128;
    const int block_n = get_group_id(0) * 128;

    // LDS storage: A is 128 rows x 16 cols. B is 16 rows x 128 cols (64 half2)
    __local half lds_A[128][16 + LDS_PAD];
    __local half2 lds_B2[16][64 + (LDS_PAD/2)];

    half2 c2[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c2[i][j] = (half2)(0.0h, 0.0h);
    }

    // 256 threads loading A (128 x 16 = 2048 halfs): each thread loads 8 halfs (vload8)
    const int load_row_a = tid >> 1; // 0..127
    const int load_col_a = (tid & 1) << 3; // 0 or 8
    const int global_row_a = block_m + load_row_a;

    // 256 threads loading B (16 x 128 = 2048 halfs): each thread loads 8 halfs (4 half2)
    const int load_row_b = tid >> 4; // 0..15
    const int load_col_b = (tid & 15) << 3; // 0, 8, 16, ..., 120
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 16;

    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 16;

        // Load A coalesced
        half8 a_val = vload8(0, &A[global_row_a * K + k_base + load_col_a]);
        vstore8(a_val, 0, &lds_A[load_row_a][load_col_a]);

        // Load B coalesced
        half8 b_val = vload8(0, &B[(k_base + load_row_b) * N + global_col_b]);
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            lds_B2[load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&b_val)[p*2], ((half*)&b_val)[p*2 + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 8 rows x 4 half2 cols (8 cols)
        const int my_m = ty * 8; // 16 * 8 = 128
        const int my_n2 = tx * 4; // 16 * 4 half2 = 64 half2 = 128 halfs

        #pragma unroll 16
        for (int k = 0; k < 16; k++) {
            half2 b_pack[4];
            b_pack[0] = lds_B2[k][my_n2 + 0];
            b_pack[1] = lds_B2[k][my_n2 + 1];
            b_pack[2] = lds_B2[k][my_n2 + 2];
            b_pack[3] = lds_B2[k][my_n2 + 3];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                half a_scalar = lds_A[my_m + i][k];
                half2 a_pack = (half2)(a_scalar, a_scalar);

                c2[i][0] = fma(a_pack, b_pack[0], c2[i][0]);
                c2[i][1] = fma(a_pack, b_pack[1], c2[i][1]);
                c2[i][2] = fma(a_pack, b_pack[2], c2[i][2]);
                c2[i][3] = fma(a_pack, b_pack[3], c2[i][3]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write back 8 rows x 8 cols
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
// Kernel: 128x128 Tile with 256 threads AND Double Buffering (Single Barrier!)
// LDS: 2 * 8.7 KB = 17.4 KB per workgroup.
// 256 threads = 4 Wave64s!
// With 17.4 KB LDS: CU hosts 3 workgroups = 12 active wavefronts (3 waves/SIMD).
// Overlaps DDR4 prefetch with 128 FMA instructions!
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_rpm_128x128_w256_double_buffer(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * 16 + tx;

    const int block_m = get_group_id(1) * 128;
    const int block_n = get_group_id(0) * 128;

    __local half lds_A[2][128][16 + LDS_PAD];
    __local half2 lds_B2[2][16][64 + (LDS_PAD/2)];

    half2 c2[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c2[i][j] = (half2)(0.0h, 0.0h);
    }

    const int load_row_a = tid >> 1;
    const int load_col_a = (tid & 1) << 3;
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 4;
    const int load_col_b = (tid & 15) << 3;
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 16;

    // Prologue: Prefetch Tile 0
    half8 a_pre = vload8(0, &A[global_row_a * K + 0 + load_col_a]);
    vstore8(a_pre, 0, &lds_A[0][load_row_a][load_col_a]);

    half8 b_pre = vload8(0, &B[(0 + load_row_b) * N + global_col_b]);
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        lds_B2[0][load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&b_pre)[p*2], ((half*)&b_pre)[p*2 + 1]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < num_tiles; t++) {
        int read_buf = t & 1;
        int write_buf = (t + 1) & 1;

        if (t + 1 < num_tiles) {
            int next_k = (t + 1) * 16;
            a_pre = vload8(0, &A[global_row_a * K + next_k + load_col_a]);
            b_pre = vload8(0, &B[(next_k + load_row_b) * N + global_col_b]);
        }

        const int my_m = ty * 8;
        const int my_n2 = tx * 4;

        #pragma unroll 16
        for (int k = 0; k < 16; k++) {
            half2 b_pack[4];
            b_pack[0] = lds_B2[read_buf][k][my_n2 + 0];
            b_pack[1] = lds_B2[read_buf][k][my_n2 + 1];
            b_pack[2] = lds_B2[read_buf][k][my_n2 + 2];
            b_pack[3] = lds_B2[read_buf][k][my_n2 + 3];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                half a_scalar = lds_A[read_buf][my_m + i][k];
                half2 a_pack = (half2)(a_scalar, a_scalar);

                c2[i][0] = fma(a_pack, b_pack[0], c2[i][0]);
                c2[i][1] = fma(a_pack, b_pack[1], c2[i][1]);
                c2[i][2] = fma(a_pack, b_pack[2], c2[i][2]);
                c2[i][3] = fma(a_pack, b_pack[3], c2[i][3]);
            }
        }

        if (t + 1 < num_tiles) {
            vstore8(a_pre, 0, &lds_A[write_buf][load_row_a][load_col_a]);
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                lds_B2[write_buf][load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&b_pre)[p*2], ((half*)&b_pre)[p*2 + 1]);
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
)CLC";

void run_test(cl_context ctx, cl_command_queue queue, cl_kernel k, const char* name, int M, int N, int K, size_t g_ws[2], size_t l_ws[2], cl_mem dA, cl_mem dB, cl_mem dC) {
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
    std::cout << "  " << std::left << std::setw(42) << name 
              << ": " << std::fixed << std::setprecision(3) << best_ms << " ms | " 
              << std::setprecision(3) << tflops << " TFLOPS\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  Experiment: 128x128 Tile with 256 Threads (4 Waves/Workgroup)  \n";
    std::cout << "=================================================================\n";

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, nullptr);

    size_t len = strlen(KERNEL_SRCS_2);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRCS_2, &len, nullptr);
    cl_int b_err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (b_err != CL_SUCCESS) {
        size_t lsz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        std::cerr << "Build Log:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel k_128 = clCreateKernel(prog, "gemm_rpm_128x128_w256", nullptr);
    cl_kernel k_128_db = clCreateKernel(prog, "gemm_rpm_128x128_w256_double_buffer", nullptr);

    int M = 2048, N = 2048, K = 2048;
    size_t bytes = (size_t)M * K * sizeof(uint16_t);
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    // Workgroup is (16, 16) = 256 threads.
    // Tile is 128 x 128.
    // Global work size: (N / 128) * 16 = N / 8.
    // (M / 128) * 16 = M / 8.
    size_t g_ws[2] = { (size_t)N / 8, (size_t)M / 8 };
    size_t l_ws[2] = { 16, 16 };

    std::cout << "\n[Experiment 1] Matrix Dimensions: 2048 x 2048 x 2048\n";
    run_test(ctx, queue, k_128, "128x128 Tile, 256 Thds (4 Waves), 2-Bar", M, N, K, g_ws, l_ws, dA, dB, dC);
    run_test(ctx, queue, k_128_db, "128x128 Tile, 256 Thds (4 Waves), 1-Bar DB", M, N, K, g_ws, l_ws, dA, dB, dC);

    int M2 = 4096;
    size_t bytes_A2 = (size_t)M2 * K * sizeof(uint16_t);
    size_t bytes_C2 = (size_t)M2 * N * sizeof(uint16_t);
    cl_mem dA2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes_A2, nullptr, nullptr);
    cl_mem dC2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_C2, nullptr, nullptr);

    size_t g_ws_2[2] = { (size_t)N / 8, (size_t)M2 / 8 };

    std::cout << "\n[Experiment 2] Large Prefill: 4096 x 2048 x 2048\n";
    run_test(ctx, queue, k_128, "128x128 Tile, 256 Thds (4 Waves), 2-Bar", M2, N, K, g_ws_2, l_ws, dA2, dB, dC2);
    run_test(ctx, queue, k_128_db, "128x128 Tile, 256 Thds (4 Waves), 1-Bar DB", M2, N, K, g_ws_2, l_ws, dA2, dB, dC2);

    return 0;
}
