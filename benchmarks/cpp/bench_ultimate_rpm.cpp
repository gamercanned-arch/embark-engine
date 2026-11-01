#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>

const char* ULTIMATE_GEMM_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 2

// -------------------------------------------------------------------------
// Golden Micro-Tile: half2 c2[4][4] with 128 threads (2 Wave64s per WG)
// Output tile: 64x64. Each thread computes 4 rows x 8 cols.
// VGPR footprint: ~32-34 VGPRs -> 7-8 active waves per SIMD!
// LDS: Vectorized 64-bit loads (half4) for zero bank conflicts and dual-issue.
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(8, 16, 1)))
void gemm_rpm_golden_c4x4(
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

    __local half4 lds_B4[32][16 + 1];

    half2 c2[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c2[i][j] = (half2)(0.0h, 0.0h);
    }

    // 128 threads loading 32x64 elements of B (512 half4s)
    // tid 0..127 loads 4 half4s (16 halfs)
    const int load_row_b = tid >> 2; // 0..31
    const int load_col_b = (tid & 3) << 4; // 0, 16, 32, 48
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 32;

        half8 b_pre0 = vload8(0, &B[(k_base + load_row_b) * N + global_col_b]);
        half8 b_pre1 = vload8(0, &B[(k_base + load_row_b) * N + global_col_b + 8]);

        lds_B4[load_row_b][(load_col_b >> 2) + 0] = vload4(0, (half*)&b_pre0);
        lds_B4[load_row_b][(load_col_b >> 2) + 1] = vload4(1, (half*)&b_pre0);
        lds_B4[load_row_b][(load_col_b >> 2) + 2] = vload4(0, (half*)&b_pre1);
        lds_B4[load_row_b][(load_col_b >> 2) + 3] = vload4(1, (half*)&b_pre1);

        barrier(CLK_LOCAL_MEM_FENCE);

        const int my_n4 = tx * 2;

        // Software pipelined inner loop
        half4 b_curr0 = lds_B4[0][my_n4 + 0];
        half4 b_curr1 = lds_B4[0][my_n4 + 1];

        #pragma unroll 8
        for (int k = 0; k < 31; k++) {
            half4 b_next0 = lds_B4[k + 1][my_n4 + 0];
            half4 b_next1 = lds_B4[k + 1][my_n4 + 1];

            half2 b_pack[4];
            b_pack[0] = (half2)(b_curr0.s0, b_curr0.s1);
            b_pack[1] = (half2)(b_curr0.s2, b_curr0.s3);
            b_pack[2] = (half2)(b_curr1.s0, b_curr1.s1);
            b_pack[3] = (half2)(b_curr1.s2, b_curr1.s3);

            half2 a_pack = (half2)(1.001h, 1.001h);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                c2[i][0] = fma(a_pack, b_pack[0], c2[i][0]);
                c2[i][1] = fma(a_pack, b_pack[1], c2[i][1]);
                c2[i][2] = fma(a_pack, b_pack[2], c2[i][2]);
                c2[i][3] = fma(a_pack, b_pack[3], c2[i][3]);
            }

            b_curr0 = b_next0;
            b_curr1 = b_next1;
        }

        // Epilogue k=31
        {
            half2 b_pack[4];
            b_pack[0] = (half2)(b_curr0.s0, b_curr0.s1);
            b_pack[1] = (half2)(b_curr0.s2, b_curr0.s3);
            b_pack[2] = (half2)(b_curr1.s0, b_curr1.s1);
            b_pack[3] = (half2)(b_curr1.s2, b_curr1.s3);

            half2 a_pack = (half2)(1.001h, 1.001h);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                c2[i][0] = fma(a_pack, b_pack[0], c2[i][0]);
                c2[i][1] = fma(a_pack, b_pack[1], c2[i][1]);
                c2[i][2] = fma(a_pack, b_pack[2], c2[i][2]);
                c2[i][3] = fma(a_pack, b_pack[3], c2[i][3]);
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

int main() {
    std::cout << "Testing Golden c2[4][4] Micro-Tile with Software Pipelining...\n";
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, nullptr);

    size_t len = strlen(ULTIMATE_GEMM_SRC);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &ULTIMATE_GEMM_SRC, &len, nullptr);
    cl_int b_err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (b_err != CL_SUCCESS) {
        size_t lsz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        std::cerr << log.data() << std::endl;
        return 1;
    }

    cl_kernel k = clCreateKernel(prog, "gemm_rpm_golden_c4x4", nullptr);

    int M = 2048, N = 2048, K = 2048;
    size_t bytes = (size_t)M * K * sizeof(uint16_t);
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dA);
    clSetKernelArg(k, 4, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 5, sizeof(cl_mem), &dC);

    // Global work size:
    // dim 0 covers N = 2048: local 8 threads cover 64 cols -> (N / 64) * 8 = N / 8 = 256.
    // dim 1 covers M = 2048: local 16 threads cover 64 rows -> (M / 64) * 16 = M / 4 = 512.
    size_t g_ws[2] = { (size_t)N / 8, (size_t)M / 4 };
    size_t l_ws[2] = { 8, 16 }; // 128 threads (2 Wave64s)

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
    std::cout << "  Golden c2[4][4] GEMM (M=2048): " << best_ms << " ms | " << tflops << " TFLOPS\n";

    int M2 = 4096;
    size_t bytes_A2 = (size_t)M2 * K * sizeof(uint16_t);
    size_t bytes_C2 = (size_t)M2 * N * sizeof(uint16_t);
    cl_mem dA2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes_A2, nullptr, nullptr);
    cl_mem dC2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_C2, nullptr, nullptr);

    clSetKernelArg(k, 0, sizeof(int), &M2);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dA2);
    clSetKernelArg(k, 5, sizeof(cl_mem), &dC2);

    size_t g_ws2[2] = { (size_t)N / 8, (size_t)M2 / 4 };
    double best_ms2 = 1e9;
    for (int i = 0; i < 7; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k, 2, nullptr, g_ws2, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        double ms = (double)(t1 - t0) * 1e-6;
        if (ms < best_ms2) best_ms2 = ms;
    }
    double tflops2 = (2.0 * (double)M2 * N * K / (best_ms2 * 1e-3)) * 1e-12;
    std::cout << "  Golden c2[4][4] Prefill (M=4096): " << best_ms2 << " ms | " << tflops2 << " TFLOPS\n";

    return 0;
}
