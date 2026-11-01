#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>

const char* KERNEL_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 2

// =========================================================================
// Kernel 1: Baseline 64x64 Tile, 64 threads (1 Wavefront)
// =========================================================================
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_rpm_64x64_1wave(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    __local half lds_A[64][32 + LDS_PAD];
    __local half2 lds_B2[32][32 + (LDS_PAD/2)];

    half2 c2[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            c2[i][j] = (half2)(0.0h, 0.0h);
        }
    }

    const int load_row_a = tid; // 0..63
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 1;        // 0..31
    const int load_col_b = (tid & 1) << 5;   // 0 or 32 halfs = 0 or 16 half2
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

// =========================================================================
// Kernel 2: 128x128 Tile, 256 threads (4 Wavefronts = 1 Wave per SIMD16)
// =========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_rpm_128x128_4waves(
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

    // LDS storage: A is [128][16 + PAD], B is [16][64 half2 + PAD]
    // K_step = 16 for minimal LDS footprint: 128*18*2 = 4.6 KB, 16*66*4 = 4.2 KB -> 8.8 KB LDS!
    __local half lds_A[128][16 + LDS_PAD];
    __local half2 lds_B2[16][64 + (LDS_PAD/2)];

    half2 c2[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            c2[i][j] = (half2)(0.0h, 0.0h);
        }
    }

    // 256 threads loading A (128 rows x 16 cols = 2048 halfs = 8 halfs per thread = half8 = 128-bit)
    // 2 threads per row: thread 2r loads cols 0..7, thread 2r+1 loads cols 8..15
    const int a_load_row = tid >> 1;        // 0..127
    const int a_load_col = (tid & 1) << 3;   // 0 or 8
    const int global_a_row = block_m + a_load_row;

    // 256 threads loading B (16 rows x 128 cols = 2048 halfs = 1024 half2 = 8 halfs per thread = 4 half2)
    // 16 threads per row:
    const int b_load_row = tid >> 4;        // 0..15
    const int b_load_col = (tid & 15) << 3;  // 0, 8, 16, ..., 120 halfs = (tid & 15) << 2 half2
    const int global_b_col = block_n + b_load_col;

    const int num_tiles = K / 16;

    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 16;

        // Perfectly coalesced 128-bit load for A
        half8 a_val = vload8(0, &A[global_a_row * K + k_base + a_load_col]);
        vstore8(a_val, 0, &lds_A[a_load_row][a_load_col]);

        // Perfectly coalesced 128-bit load for B
        half8 b_raw = vload8(0, &B[(k_base + b_load_row) * N + global_b_col]);
        const int b2_col = (tid & 15) << 2;
        lds_B2[b_load_row][b2_col + 0] = (half2)(((half*)&b_raw)[0], ((half*)&b_raw)[1]);
        lds_B2[b_load_row][b2_col + 1] = (half2)(((half*)&b_raw)[2], ((half*)&b_raw)[3]);
        lds_B2[b_load_row][b2_col + 2] = (half2)(((half*)&b_raw)[4], ((half*)&b_raw)[5]);
        lds_B2[b_load_row][b2_col + 3] = (half2)(((half*)&b_raw)[6], ((half*)&b_raw)[7]);

        barrier(CLK_LOCAL_MEM_FENCE);

        const int my_m = ty * 8;
        const int my_n2 = tx * 4;

        #pragma unroll 16
        for (int k = 0; k < 16; k++) {
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

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int out_row = block_m + ty * 8 + i;
        if (out_row < M && block_n + tx * 8 < N) {
            __global half* out_ptr = C + out_row * N + block_n + tx * 8;
            half8 out_v = (half8)(c2[i][0].x, c2[i][0].y, c2[i][1].x, c2[i][1].y,
                                 c2[i][2].x, c2[i][2].y, c2[i][3].x, c2[i][3].y);
            vstore8(out_v, 0, out_ptr);
        }
    }
}

// =========================================================================
// Kernel 3: 128x128 Tile with 2D Grid Panel Swizzling for 512 KB L2 Cache
// =========================================================================
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_rpm_128x128_swizzled(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * 16 + tx;

    // Panel Swizzling: Group tiles into stripes of width 4 tiles (512 cols)
    const int grid_n = get_num_groups(0);
    const int grid_m = get_num_groups(1);
    const int linear_gid = get_group_id(1) * grid_n + get_group_id(0);

    const int PANEL_W = 4;
    const int panel_idx = linear_gid / (grid_m * PANEL_W);
    const int rem = linear_gid % (grid_m * PANEL_W);
    const int swizzled_n_local = rem % PANEL_W;
    const int swizzled_m = rem / PANEL_W;
    const int swizzled_n = panel_idx * PANEL_W + swizzled_n_local;

    const int block_m = swizzled_m * 128;
    const int block_n = swizzled_n * 128;

    __local half lds_A[128][16 + LDS_PAD];
    __local half2 lds_B2[16][64 + (LDS_PAD/2)];

    half2 c2[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            c2[i][j] = (half2)(0.0h, 0.0h);
        }
    }

    const int a_load_row = tid >> 1;
    const int a_load_col = (tid & 1) << 3;
    const int global_a_row = block_m + a_load_row;

    const int b_load_row = tid >> 4;
    const int b_load_col = (tid & 15) << 3;
    const int global_b_col = block_n + b_load_col;

    const int num_tiles = K / 16;

    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 16;

        half8 a_val = vload8(0, &A[global_a_row * K + k_base + a_load_col]);
        vstore8(a_val, 0, &lds_A[a_load_row][a_load_col]);

        half8 b_raw = vload8(0, &B[(k_base + b_load_row) * N + global_b_col]);
        const int b2_col = (tid & 15) << 2;
        lds_B2[b_load_row][b2_col + 0] = (half2)(((half*)&b_raw)[0], ((half*)&b_raw)[1]);
        lds_B2[b_load_row][b2_col + 1] = (half2)(((half*)&b_raw)[2], ((half*)&b_raw)[3]);
        lds_B2[b_load_row][b2_col + 2] = (half2)(((half*)&b_raw)[4], ((half*)&b_raw)[5]);
        lds_B2[b_load_row][b2_col + 3] = (half2)(((half*)&b_raw)[6], ((half*)&b_raw)[7]);

        barrier(CLK_LOCAL_MEM_FENCE);

        const int my_m = ty * 8;
        const int my_n2 = tx * 4;

        #pragma unroll 16
        for (int k = 0; k < 16; k++) {
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

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int out_row = block_m + ty * 8 + i;
        if (out_row < M && block_n + tx * 8 < N) {
            __global half* out_ptr = C + out_row * N + block_n + tx * 8;
            half8 out_v = (half8)(c2[i][0].x, c2[i][0].y, c2[i][1].x, c2[i][1].y,
                                 c2[i][2].x, c2[i][2].y, c2[i][3].x, c2[i][3].y);
            vstore8(out_v, 0, out_ptr);
        }
    }
}
)CLC";

void run_test(cl_context ctx, cl_command_queue queue, cl_kernel k, const std::string& name,
              int M, int N, int K, size_t wg_x, size_t wg_y, size_t tile_m, size_t tile_n,
              cl_mem dA, cl_mem dB, cl_mem dC)
{
    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dA);
    clSetKernelArg(k, 4, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 5, sizeof(cl_mem), &dC);

    size_t g_sz[2] = { (size_t)N / tile_n * wg_x, (size_t)M / tile_m * wg_y };
    size_t l_sz[2] = { wg_x, wg_y };

    // Warmup
    clEnqueueNDRangeKernel(queue, k, 2, nullptr, g_sz, l_sz, 0, nullptr, nullptr);
    clFinish(queue);

    double best_tflops = 0.0;
    double min_ms = 1e9;

    for (int r = 0; r < 5; r++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k, 2, nullptr, g_sz, l_sz, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);

        double ns = (double)(t1 - t0);
        double ms = ns / 1e6;
        double flops = 2.0 * (double)M * N * K;
        double tflops = (flops / ns) * 1e-3;

        if (ms < min_ms) min_ms = ms;
        if (tflops > best_tflops) best_tflops = tflops;
    }

    std::cout << std::left << std::setw(36) << name 
              << " | " << std::right << std::setw(8) << std::fixed << std::setprecision(3) << min_ms << " ms"
              << " | " << std::setw(6) << std::setprecision(3) << best_tflops << " TFLOPS"
              << " (" << std::setprecision(1) << (best_tflops / 4.096 * 100.0) << "% peak)\n";
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "AMD Vega 8 GEMM & Tile Geometry Empirical Architect Benchmark\n";
    std::cout << "Target: gfx90c (8 CUs, 512 ALUs @ 2000 MHz = 4.096 TFLOPS FP16 RPM)\n";
    std::cout << "====================================================================\n\n";

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, nullptr);

    size_t len = strlen(KERNEL_SRC);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, &len, nullptr);
    cl_int b_err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (b_err != CL_SUCCESS) {
        size_t lsz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        std::cerr << "Build Log:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel k1 = clCreateKernel(prog, "gemm_rpm_64x64_1wave", nullptr);
    cl_kernel k2 = clCreateKernel(prog, "gemm_rpm_128x128_4waves", nullptr);
    cl_kernel k3 = clCreateKernel(prog, "gemm_rpm_128x128_swizzled", nullptr);

    struct TestConfig { int M; int N; int K; const char* desc; };
    std::vector<TestConfig> configs = {
        { 2048, 2048, 2048, "Square 2048x2048x2048" },
        { 4096, 2048, 2048, "Prefill M=4096 (4096x2048x2048)" },
        { 2816, 3072, 3072, "DiT Denoise (2816x3072x3072)" }
    };

    for (const auto& cfg : configs) {
        std::cout << "\n--- Workload: " << cfg.desc << " ---\n";
        size_t bytes_a = (size_t)cfg.M * cfg.K * sizeof(uint16_t);
        size_t bytes_b = (size_t)cfg.K * cfg.N * sizeof(uint16_t);
        size_t bytes_c = (size_t)cfg.M * cfg.N * sizeof(uint16_t);

        cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes_a, nullptr, nullptr);
        cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes_b, nullptr, nullptr);
        cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_c, nullptr, nullptr);

        run_test(ctx, queue, k1, "64x64 (1 Wave, 64 thr, Raster)", cfg.M, cfg.N, cfg.K, 8, 8, 64, 64, dA, dB, dC);
        run_test(ctx, queue, k2, "128x128 (4 Waves, 256 thr, Raster)", cfg.M, cfg.N, cfg.K, 16, 16, 128, 128, dA, dB, dC);
        run_test(ctx, queue, k3, "128x128 (4 Waves, Swizzled 4-Panel)", cfg.M, cfg.N, cfg.K, 16, 16, 128, 128, dA, dB, dC);

        clReleaseMemObject(dA);
        clReleaseMemObject(dB);
        clReleaseMemObject(dC);
    }

    clReleaseKernel(k1);
    clReleaseKernel(k2);
    clReleaseKernel(k3);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
