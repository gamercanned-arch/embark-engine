#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>

const char* RPM_GEMM_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 2 // 34 halfs = 17 dwords (17 % 32 != 0, prime to bank count)

// True Rapid Packed Math (RPM) GEMM for AMD Vega 8 (gfx90c)
// Workgroup: (8, 8, 1) = 64 threads (Wave64)
// Tile: 64x64 output
// Micro-tile per thread: 8 rows x 4 half2 cols = 8x8 halfs
// Inner loop uses native half2 fma -> lowers directly to v_pk_fma_f16!
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_rpm_wave64(
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

    // LDS storage: A is [64][32 + PAD], B is [32][32 half2 + PAD]
    __local half lds_A[64][32 + LDS_PAD];
    __local half2 lds_B2[32][32 + (LDS_PAD/2)];

    // 8 rows x 4 half2 cols = 32 registers (down from 64!)
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

        // Load A coalesced (32 halfs per row)
        half16 a0 = vload16(0, &A[global_row_a * K + k_base + 0]);
        half16 a1 = vload16(0, &A[global_row_a * K + k_base + 16]);
        vstore16(a0, 0, &lds_A[load_row_a][0]);
        vstore16(a1, 0, &lds_A[load_row_a][16]);

        // Load B coalesced (32 halfs = 16 half2)
        half16 b_raw = vload16(0, &B[(k_base + load_row_b) * N + global_col_b]);
        half16 b_raw2 = vload16(0, &B[(k_base + load_row_b) * N + global_col_b + 16]);

        #pragma unroll
        for (int p = 0; p < 8; p++) {
            lds_B2[load_row_b][(load_col_b >> 1) + p] = (half2)(((half*)&b_raw)[p*2], ((half*)&b_raw)[p*2 + 1]);
            lds_B2[load_row_b][(load_col_b >> 1) + 8 + p] = (half2)(((half*)&b_raw2)[p*2], ((half*)&b_raw2)[p*2 + 1]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 8 rows x 4 half2 cols using Rapid Packed Math v_pk_fma_f16
        const int my_m = ty * 8;
        const int my_n2 = tx * 4; // 4 half2 = 8 halfs

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

    // Write back 8x8 results
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

int main() {
    std::cout << "Testing True Rapid Packed Math (v_pk_fma_f16) GEMM on Vega 8...\n";
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, nullptr);

    size_t len = strlen(RPM_GEMM_SRC);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &RPM_GEMM_SRC, &len, nullptr);
    cl_int b_err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (b_err != CL_SUCCESS) {
        size_t lsz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        std::cerr << log.data() << std::endl;
        return 1;
    }

    cl_kernel k_rpm = clCreateKernel(prog, "gemm_rpm_wave64", nullptr);

    int M = 2048, N = 2048, K = 2048;
    size_t bytes = (size_t)M * K * sizeof(uint16_t);
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);

    clSetKernelArg(k_rpm, 0, sizeof(int), &M);
    clSetKernelArg(k_rpm, 1, sizeof(int), &N);
    clSetKernelArg(k_rpm, 2, sizeof(int), &K);
    clSetKernelArg(k_rpm, 3, sizeof(cl_mem), &dA);
    clSetKernelArg(k_rpm, 4, sizeof(cl_mem), &dB);
    clSetKernelArg(k_rpm, 5, sizeof(cl_mem), &dC);

    size_t g_ws[2] = { (size_t)N / 8, (size_t)M / 8 };
    size_t l_ws[2] = { 8, 8 };

    // Warmup
    clEnqueueNDRangeKernel(queue, k_rpm, 2, nullptr, g_ws, l_ws, 0, nullptr, nullptr);
    clFinish(queue);

    std::cout << "\nBenchmarking gemm_rpm_wave64 (M=2048, N=2048, K=2048):\n";
    for (int i = 0; i < 5; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_rpm, 2, nullptr, g_ws, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        double ms = (double)(t1 - t0) * 1e-6;
        double tflops = (2.0 * M * N * K / (ms * 1e-3)) * 1e-12;
        std::cout << "  RPM Run " << i << ": " << std::fixed << std::setprecision(3) << ms << " ms | " 
                  << std::setprecision(3) << tflops << " TFLOPS\n";
    }

    // Also test M=1024 and M=4096
    int m_tests[] = { 512, 1024, 4096 };
    for (int test_m : m_tests) {
        size_t bytes_A = (size_t)test_m * K * sizeof(uint16_t);
        size_t bytes_C = (size_t)test_m * N * sizeof(uint16_t);
        cl_mem tA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes_A, nullptr, nullptr);
        cl_mem tC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_C, nullptr, nullptr);

        clSetKernelArg(k_rpm, 0, sizeof(int), &test_m);
        clSetKernelArg(k_rpm, 3, sizeof(cl_mem), &tA);
        clSetKernelArg(k_rpm, 5, sizeof(cl_mem), &tC);

        size_t cur_g_ws[2] = { (size_t)N / 8, (size_t)test_m / 8 };

        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_rpm, 2, nullptr, cur_g_ws, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        double ms = (double)(t1 - t0) * 1e-6;
        double tflops = (2.0 * (double)test_m * N * K / (ms * 1e-3)) * 1e-12;
        std::cout << "  Prefill M=" << test_m << " (K=2048, N=2048): " << ms << " ms | " << tflops << " TFLOPS\n";

        clReleaseMemObject(tA);
        clReleaseMemObject(tC);
    }

    return 0;
}
