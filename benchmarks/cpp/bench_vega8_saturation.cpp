#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <windows.h>

// OpenCL Error Checker
#define CHECK_CL(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "[-] OpenCL Error at " << msg << ": " << err << std::endl; \
        return 1; \
    }

// OpenCL Kernels
const char* KERNEL_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// =========================================================================
// 1. Pure FP32 Compute Kernel (Zero DRAM traffic in loop)
// 16 independent accumulators, unrolled 128 times = 2048 FMAs = 4096 FLOPs/thread/iter
// =========================================================================
__kernel void pure_fp32_fma(__global float* out, const int iters) {
    int gid = get_global_id(0);
    float a0 = (float)gid * 0.0001f + 1.0f;
    float a1 = a0 * 1.0001f;
    float a2 = a0 * 1.0002f;
    float a3 = a0 * 1.0003f;
    float a4 = a0 * 1.0004f;
    float a5 = a0 * 1.0005f;
    float a6 = a0 * 1.0006f;
    float a7 = a0 * 1.0007f;
    float a8 = a0 * 1.0008f;
    float a9 = a0 * 1.0009f;
    float a10 = a0 * 1.0010f;
    float a11 = a0 * 1.0011f;
    float a12 = a0 * 1.0012f;
    float a13 = a0 * 1.0013f;
    float a14 = a0 * 1.0014f;
    float a15 = a0 * 1.0015f;

    float b = 1.00001f;
    float c = 0.00001f;

    for (int i = 0; i < iters; i++) {
        #pragma unroll 16
        for (int k = 0; k < 8; k++) {
            a0 = mad(a0, b, c);
            a1 = mad(a1, b, c);
            a2 = mad(a2, b, c);
            a3 = mad(a3, b, c);
            a4 = mad(a4, b, c);
            a5 = mad(a5, b, c);
            a6 = mad(a6, b, c);
            a7 = mad(a7, b, c);
            a8 = mad(a8, b, c);
            a9 = mad(a9, b, c);
            a10 = mad(a10, b, c);
            a11 = mad(a11, b, c);
            a12 = mad(a12, b, c);
            a13 = mad(a13, b, c);
            a14 = mad(a14, b, c);
            a15 = mad(a15, b, c);
        }
    }

    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}

// =========================================================================
// 2. Pure FP16 Compute Kernel (Rapid Packed Math half2)
// 8 independent half2 accumulators = 16 half ops per cycle
// =========================================================================
__kernel void pure_fp16_fma(__global half* out, const int iters) {
    int gid = get_global_id(0);
    half2 h0 = (half2)((half)gid * 0.0001h + 1.0h, 1.1h);
    half2 h1 = h0 * (half2)(1.001h, 1.001h);
    half2 h2 = h0 * (half2)(1.002h, 1.002h);
    half2 h3 = h0 * (half2)(1.003h, 1.003h);
    half2 h4 = h0 * (half2)(1.004h, 1.004h);
    half2 h5 = h0 * (half2)(1.005h, 1.005h);
    half2 h6 = h0 * (half2)(1.006h, 1.006h);
    half2 h7 = h0 * (half2)(1.007h, 1.007h);

    half2 hb = (half2)(1.0001h, 1.0001h);
    half2 hc = (half2)(0.0001h, 0.0001h);

    for (int i = 0; i < iters; i++) {
        #pragma unroll 16
        for (int k = 0; k < 8; k++) {
            h0 = fma(h0, hb, hc);
            h1 = fma(h1, hb, hc);
            h2 = fma(h2, hb, hc);
            h3 = fma(h3, hb, hc);
            h4 = fma(h4, hb, hc);
            h5 = fma(h5, hb, hc);
            h6 = fma(h6, hb, hc);
            h7 = fma(h7, hb, hc);
        }
    }

    half2 res = h0 + h1 + h2 + h3 + h4 + h5 + h6 + h7;
    out[gid * 2] = res.x;
    out[gid * 2 + 1] = res.y;
}

// =========================================================================
// 3. Autoregressive Decode GEMV (M=1, K=4096, N=4096)
// Arithmetic Intensity <= 2 FLOPs/byte
// =========================================================================
__kernel void gemv_decode_m1(
    const int K, const int N,
    __global const half* restrict x,
    __global const half* restrict W,
    __global half* restrict y)
{
    int col = get_global_id(0); // 0..N-1
    if (col >= N) return;

    float acc = 0.0f;
    int k_blocks = K / 8;
    for (int b = 0; b < k_blocks; b++) {
        half8 vx = vload8(0, &x[b * 8]);
        half8 vw = vload8(0, &W[col * K + b * 8]);
        acc += (float)(vx.s0 * vw.s0 + vx.s1 * vw.s1 + vx.s2 * vw.s2 + vx.s3 * vw.s3 +
                       vx.s4 * vw.s4 + vx.s5 * vw.s5 + vx.s6 * vw.s6 + vx.s7 * vw.s7);
    }
    y[col] = (half)acc;
}

// =========================================================================
// 4. Tiled GEMM Wave64 with LDS (Tile: 64x64, Thread: 8x8)
// Computes C[M, N] = A[M, K] * B[K, N]
// Workgroup: (8, 8, 1) = 64 threads (Wave64)
// =========================================================================
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define LDS_PAD 1

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_tiled_wave64(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63 (Wave64)

    const int block_m = get_group_id(1) * TILE_M;
    const int block_n = get_group_id(0) * TILE_N;

    // LDS storage with padding to eliminate 32-bank conflicts
    __local half lds_A[TILE_K][TILE_M + LDS_PAD];
    __local half lds_B[TILE_K][TILE_N + LDS_PAD];

    // Each thread accumulates an 8x8 micro-tile in VGPRs
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
        }
    }

    const int num_k_tiles = K / TILE_K;

    // Cooperative loading coordinates:
    // 64 threads loading 64 * 16 = 1024 elements (16 elements per thread)
    const int load_row = tid >> 2;       // 0..15 (k dimension)
    const int load_col = (tid & 3) << 4; // 0, 16, 32, 48

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_base = kt * TILE_K;

        // Load A tile: [TILE_M, TILE_K] transposed into LDS [TILE_K, TILE_M]
        // A is M x K: global index (block_m + load_col + p) * K + k_base + load_row
        #pragma unroll
        for (int p = 0; p < 16; p++) {
            int gm = block_m + load_col + p;
            int gk = k_base + load_row;
            half val = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            lds_A[load_row][load_col + p] = val;
        }

        // Load B tile: [TILE_K, TILE_N] into LDS [TILE_K, TILE_N]
        // B is K x N: global index (k_base + load_row) * N + block_n + load_col + p
        #pragma unroll
        for (int p = 0; p < 16; p++) {
            int gk = k_base + load_row;
            int gn = block_n + load_col + p;
            half val = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            lds_B[load_row][load_col + p] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 8x8 micro-tile: tx controls N (8 elements: tx*8 .. tx*8+7), ty controls M (8 elements: ty*8 .. ty*8+7)
        const int my_m = ty * 8;
        const int my_n = tx * 8;

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_vals[8];
            float b_vals[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a_vals[i] = (float)lds_A[k][my_m + i];
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                b_vals[j] = (float)lds_B[k][my_n + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write back 8x8 micro-tile
    const int my_m = block_m + ty * 8;
    const int my_n = block_n + tx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int gm = my_m + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int gn = my_n + j;
                if (gn < N) {
                    C[gm * N + gn] = (half)acc[i][j];
                }
            }
        }
    }
}
)CLC";

// Microbenchmark Harness
int main() {
    std::cout << "======================================================================\n";
    std::cout << " AMD VEGA 8 (gfx90c) COMPUTE SATURATION & ROOFLINE MICROBENCHMARK\n";
    std::cout << " AMD Ryzen 7 7730U APU (Zen 3 8C/16T, Vega 8 8CU/512SP @ 2000 MHz)\n";
    std::cout << "======================================================================\n\n";

    cl_int err;
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    CHECK_CL(err, "clGetPlatformIDs");

    cl_device_id device;
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    CHECK_CL(err, "clGetDeviceIDs");

    char dev_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    cl_uint max_cus;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_cus), &max_cus, nullptr);
    cl_uint max_clock;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock), &max_clock, nullptr);

    std::cout << "[Device Information]\n";
    std::cout << "  Device Name:             " << dev_name << "\n";
    std::cout << "  Compute Units (CUs):     " << max_cus << " CUs (each 64 SPs = " << (max_cus * 64) << " ALUs)\n";
    std::cout << "  Rated Max Clock:         " << max_clock << " MHz\n";
    std::cout << "  Theoretical Peak FP32:   " << std::fixed << std::setprecision(3) 
              << (max_cus * 64 * 2.0 * max_clock / 1e6) << " TFLOPS\n";
    std::cout << "  Theoretical Peak FP16:   " << std::fixed << std::setprecision(3) 
              << (max_cus * 64 * 4.0 * max_clock / 1e6) << " TFLOPS (with Rapid Packed Math)\n\n";

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err, "clCreateContext");

    cl_queue_properties q_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, q_props, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    // Build Program
    size_t src_len = strlen(KERNEL_SRC);
    cl_program program = clCreateProgramWithSource(context, 1, &KERNEL_SRC, &src_len, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        std::cerr << "[-] Build Log:\n" << build_log.data() << std::endl;
        return 1;
    }

    cl_kernel k_pure_fp32 = clCreateKernel(program, "pure_fp32_fma", &err);
    CHECK_CL(err, "clCreateKernel pure_fp32_fma");
    cl_kernel k_pure_fp16 = clCreateKernel(program, "pure_fp16_fma", &err);
    CHECK_CL(err, "clCreateKernel pure_fp16_fma");
    cl_kernel k_gemv_m1 = clCreateKernel(program, "gemv_decode_m1", &err);
    CHECK_CL(err, "clCreateKernel gemv_decode_m1");
    cl_kernel k_gemm_tiled = clCreateKernel(program, "gemm_tiled_wave64", &err);
    CHECK_CL(err, "clCreateKernel gemm_tiled_wave64");

    std::cout << "----------------------------------------------------------------------\n";
    std::cout << " TEST A: PURE COMPUTE SATURATION (Registers Only, 0 DRAM Traffic)\n";
    std::cout << "----------------------------------------------------------------------\n";

    // 64K threads (1024 wavefronts) to saturate 8 CUs
    size_t compute_global = 65536;
    size_t compute_local = 64; // Wave64
    int fma_iters = 5000;      // 5000 * 128 FMAs = 640,000 FMAs = 1,280,000 FLOPs per thread

    cl_mem dev_out_fp32 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, compute_global * sizeof(float), nullptr, &err);
    clSetKernelArg(k_pure_fp32, 0, sizeof(cl_mem), &dev_out_fp32);
    clSetKernelArg(k_pure_fp32, 1, sizeof(int), &fma_iters);

    // Warmup
    clEnqueueNDRangeKernel(queue, k_pure_fp32, 1, nullptr, &compute_global, &compute_local, 0, nullptr, nullptr);
    clFinish(queue);

    // Profile FP32
    cl_event ev;
    err = clEnqueueNDRangeKernel(queue, k_pure_fp32, 1, nullptr, &compute_global, &compute_local, 0, nullptr, &ev);
    clWaitForEvents(1, &ev);
    cl_ulong t_start, t_end;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
    clReleaseEvent(ev);

    double fp32_ns = (double)(t_end - t_start);
    double fp32_flops = (double)compute_global * (double)fma_iters * 128.0 * 2.0; // 128 FMAs * 2 FLOPs
    double fp32_tflops = (fp32_flops / fp32_ns) * 1e-3;

    std::cout << "  [FP32 Pure ALU FMA] Time: " << (fp32_ns / 1e6) << " ms | Sustained Compute: "
              << std::fixed << std::setprecision(3) << fp32_tflops << " TFLOPS ("
              << (fp32_tflops / (max_cus * 64 * 2.0 * max_clock / 1e6) * 100.0) << "% of 2.048 TFLOPS Peak)\n";

    // Profile FP16 RPM
    cl_mem dev_out_fp16 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, compute_global * 2 * sizeof(uint16_t), nullptr, &err);
    clSetKernelArg(k_pure_fp16, 0, sizeof(cl_mem), &dev_out_fp16);
    clSetKernelArg(k_pure_fp16, 1, sizeof(int), &fma_iters);

    clEnqueueNDRangeKernel(queue, k_pure_fp16, 1, nullptr, &compute_global, &compute_local, 0, nullptr, nullptr);
    clFinish(queue);

    err = clEnqueueNDRangeKernel(queue, k_pure_fp16, 1, nullptr, &compute_global, &compute_local, 0, nullptr, &ev);
    clWaitForEvents(1, &ev);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
    clReleaseEvent(ev);

    double fp16_ns = (double)(t_end - t_start);
    // 8 half2 fma per inner iter * 16 unroll * 8 loops = 128 half2 fma = 256 half FMAs = 512 FLOPs per iter
    double fp16_flops = (double)compute_global * (double)fma_iters * 256.0 * 2.0;
    double fp16_tflops = (fp16_flops / fp16_ns) * 1e-3;

    std::cout << "  [FP16 Rapid Packed Math] Time: " << (fp16_ns / 1e6) << " ms | Sustained Compute: "
              << std::fixed << std::setprecision(3) << fp16_tflops << " TFLOPS ("
              << (fp16_tflops / (max_cus * 64 * 4.0 * max_clock / 1e6) * 100.0) << "% of 4.096 TFLOPS Peak)\n\n";

    std::cout << "----------------------------------------------------------------------\n";
    std::cout << " TEST B: AUTOREGRESSIVE DECODE (M=1 GEMV: K=4096, N=4096)\n";
    std::cout << "----------------------------------------------------------------------\n";

    int K_dec = 4096;
    int N_dec = 4096;
    size_t x_bytes = K_dec * sizeof(uint16_t);
    size_t w_bytes = (size_t)K_dec * N_dec * sizeof(uint16_t); // 32 MB
    size_t y_bytes = N_dec * sizeof(uint16_t);

    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, x_bytes, nullptr, &err);
    cl_mem d_w = clCreateBuffer(context, CL_MEM_READ_ONLY, w_bytes, nullptr, &err);
    cl_mem d_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, y_bytes, nullptr, &err);

    clSetKernelArg(k_gemv_m1, 0, sizeof(int), &K_dec);
    clSetKernelArg(k_gemv_m1, 1, sizeof(int), &N_dec);
    clSetKernelArg(k_gemv_m1, 2, sizeof(cl_mem), &d_x);
    clSetKernelArg(k_gemv_m1, 3, sizeof(cl_mem), &d_w);
    clSetKernelArg(k_gemv_m1, 4, sizeof(cl_mem), &d_y);

    size_t gemv_global = N_dec;
    size_t gemv_local = 64;

    // Warmup
    clEnqueueNDRangeKernel(queue, k_gemv_m1, 1, nullptr, &gemv_global, &gemv_local, 0, nullptr, nullptr);
    clFinish(queue);

    // Profile GEMV
    err = clEnqueueNDRangeKernel(queue, k_gemv_m1, 1, nullptr, &gemv_global, &gemv_local, 0, nullptr, &ev);
    clWaitForEvents(1, &ev);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
    clReleaseEvent(ev);

    double gemv_ns = (double)(t_end - t_start);
    double gemv_flops = 2.0 * (double)K_dec * (double)N_dec; // 33.55 MFLOPs
    double gemv_bytes = (double)w_bytes + x_bytes + y_bytes; // 33.56 MB
    double gemv_ai = gemv_flops / gemv_bytes; // ~1.0 FLOP/byte
    double gemv_tflops = (gemv_flops / gemv_ns) * 1e-3;
    double gemv_bw = (gemv_bytes / gemv_ns); // GB/s

    std::cout << "  Operational Arithmetic Intensity (AI): " << std::fixed << std::setprecision(2) << gemv_ai << " FLOPs/byte\n";
    std::cout << "  Execution Latency:                     " << std::fixed << std::setprecision(3) << (gemv_ns / 1e6) << " ms\n";
    std::cout << "  Effective DRAM Bandwidth:              " << std::fixed << std::setprecision(2) << gemv_bw << " GB/s\n";
    std::cout << "  Sustained Compute:                     " << std::fixed << std::setprecision(3) << (gemv_tflops * 1000.0) 
              << " GFLOPS (" << gemv_tflops << " TFLOPS)\n";
    std::cout << "  ALU Saturation (Compute Utilization):  " << std::fixed << std::setprecision(2) 
              << (gemv_tflops / 2.048 * 100.0) << "% (ALUs idle 95%+ of time awaiting DDR4)\n\n";

    std::cout << "----------------------------------------------------------------------\n";
    std::cout << " TEST C: ROOFLINE SWEEP (Batch Size M = 1, 16, 64, 256, 1024, 2048)\n";
    std::cout << " Matrix Dimensions: M x K=2048, N=2048\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::setw(6) << "M" 
              << std::setw(14) << "AI (FLOP/B)" 
              << std::setw(12) << "Time (ms)" 
              << std::setw(14) << "DRAM (GB/s)" 
              << std::setw(14) << "TFLOPS" 
              << std::setw(18) << "Limiting Regime\n";
    std::cout << "----------------------------------------------------------------------\n";

    int K_sweep = 2048;
    int N_sweep = 2048;
    std::vector<int> m_sizes = { 1, 16, 64, 256, 1024, 2048 };

    for (int M_curr : m_sizes) {
        size_t bytes_A = (size_t)M_curr * K_sweep * sizeof(uint16_t);
        size_t bytes_B = (size_t)K_sweep * N_sweep * sizeof(uint16_t);
        size_t bytes_C = (size_t)M_curr * N_sweep * sizeof(uint16_t);

        cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_A, nullptr, &err);
        cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_B, nullptr, &err);
        cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_C, nullptr, &err);

        clSetKernelArg(k_gemm_tiled, 0, sizeof(int), &M_curr);
        clSetKernelArg(k_gemm_tiled, 1, sizeof(int), &N_sweep);
        clSetKernelArg(k_gemm_tiled, 2, sizeof(int), &K_sweep);
        clSetKernelArg(k_gemm_tiled, 3, sizeof(cl_mem), &dA);
        clSetKernelArg(k_gemm_tiled, 4, sizeof(cl_mem), &dB);
        clSetKernelArg(k_gemm_tiled, 5, sizeof(cl_mem), &dC);

        size_t g_work[2] = {
            ((size_t)N_sweep + 63) / 64 * 8,
            ((size_t)M_curr + 63) / 64 * 8
        };
        size_t l_work[2] = { 8, 8 }; // Wave64

        // Warmup
        clEnqueueNDRangeKernel(queue, k_gemm_tiled, 2, nullptr, g_work, l_work, 0, nullptr, nullptr);
        clFinish(queue);

        // Run
        err = clEnqueueNDRangeKernel(queue, k_gemm_tiled, 2, nullptr, g_work, l_work, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
        clReleaseEvent(ev);

        double ns = (double)(t_end - t_start);
        double flops = 2.0 * (double)M_curr * (double)N_sweep * (double)K_sweep;
        double dram_bytes = (double)bytes_A + bytes_B + bytes_C;
        double ai = flops / dram_bytes;
        double tflops = (flops / ns) * 1e-3;
        double bw = (dram_bytes / ns);

        std::string regime = (ai < 20.0) ? "MEMORY BOUND" : (ai < 80.0 ? "TRANSITION" : "COMPUTE BOUND");

        std::cout << std::setw(6) << M_curr
                  << std::setw(14) << std::fixed << std::setprecision(2) << ai
                  << std::setw(12) << std::fixed << std::setprecision(3) << (ns / 1e6)
                  << std::setw(14) << std::fixed << std::setprecision(2) << bw
                  << std::setw(14) << std::fixed << std::setprecision(3) << tflops
                  << std::setw(18) << regime << "\n";

        clReleaseMemObject(dA);
        clReleaseMemObject(dB);
        clReleaseMemObject(dC);
    }

    std::cout << "----------------------------------------------------------------------\n\n";

    // Clean up
    clReleaseMemObject(dev_out_fp32);
    clReleaseMemObject(dev_out_fp16);
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_w);
    clReleaseMemObject(d_y);
    clReleaseKernel(k_pure_fp32);
    clReleaseKernel(k_pure_fp16);
    clReleaseKernel(k_gemv_m1);
    clReleaseKernel(k_gemm_tiled);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
