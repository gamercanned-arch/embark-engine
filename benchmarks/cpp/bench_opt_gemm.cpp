#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>

const char* OPT_GEMM_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Coalesced, High-Occupancy FP32/FP16 GEMM on Vega 8 (gfx90c)
// Workgroup: (16, 16) = 256 threads (4 Wave64s per workgroup for latency hiding!)
// Tile: 128x128 output, each thread computes 8x8 micro-tile in FP32
// K-step: 16
// All global loads are strictly vector vload4 / vload8 COALESCED.

#define TS_M 64
#define TS_N 64
#define TS_K 16
#define LDS_P 1

__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_coalesced_wave64(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C)
{
    const int tx = get_local_id(0); // 0..7
    const int ty = get_local_id(1); // 0..7
    const int tid = ty * 8 + tx;    // 0..63

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    __local float lds_A[16][64 + LDS_P];
    __local float lds_B[16][64 + LDS_P];

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // 64 threads loading 64x16 = 1024 floats (16 floats per thread)
    // To ensure coalescing: consecutive threads load consecutive elements along N or K
    // A: 64 rows, 16 cols. tid 0..63 loads row (tid), cols 0..15 in 4 vector loads
    int a_row = tid; // 0..63
    int b_row = tid >> 2; // 0..15
    int b_col = (tid & 3) << 4; // 0, 16, 32, 48

    for (int kt = 0; kt < K; kt += 16) {
        // Load A coalesced: thread loads row (block_m + a_row)
        float4 a0 = vload4(0, &A[(block_m + a_row) * K + kt + 0]);
        float4 a1 = vload4(0, &A[(block_m + a_row) * K + kt + 4]);
        float4 a2 = vload4(0, &A[(block_m + a_row) * K + kt + 8]);
        float4 a3 = vload4(0, &A[(block_m + a_row) * K + kt + 12]);

        lds_A[0][a_row] = a0.x; lds_A[1][a_row] = a0.y; lds_A[2][a_row] = a0.z; lds_A[3][a_row] = a0.w;
        lds_A[4][a_row] = a1.x; lds_A[5][a_row] = a1.y; lds_A[6][a_row] = a1.z; lds_A[7][a_row] = a1.w;
        lds_A[8][a_row] = a2.x; lds_A[9][a_row] = a2.y; lds_A[10][a_row] = a2.z; lds_A[11][a_row] = a2.w;
        lds_A[12][a_row] = a3.x; lds_A[13][a_row] = a3.y; lds_A[14][a_row] = a3.z; lds_A[15][a_row] = a3.w;

        // Load B coalesced: 16 rows, 64 cols
        float4 b0 = vload4(0, &B[(kt + b_row) * N + block_n + b_col + 0]);
        float4 b1 = vload4(0, &B[(kt + b_row) * N + block_n + b_col + 4]);
        float4 b2 = vload4(0, &B[(kt + b_row) * N + block_n + b_col + 8]);
        float4 b3 = vload4(0, &B[(kt + b_row) * N + block_n + b_col + 12]);

        lds_B[b_row][b_col + 0] = b0.x; lds_B[b_row][b_col + 1] = b0.y; lds_B[b_row][b_col + 2] = b0.z; lds_B[b_row][b_col + 3] = b0.w;
        lds_B[b_row][b_col + 4] = b1.x; lds_B[b_row][b_col + 5] = b1.y; lds_B[b_row][b_col + 6] = b1.z; lds_B[b_row][b_col + 7] = b1.w;
        lds_B[b_row][b_col + 8] = b2.x; lds_B[b_row][b_col + 9] = b2.y; lds_B[b_row][b_col + 10] = b2.z; lds_B[b_row][b_col + 11] = b2.w;
        lds_B[b_row][b_col + 12] = b3.x; lds_B[b_row][b_col + 13] = b3.y; lds_B[b_row][b_col + 14] = b3.z; lds_B[b_row][b_col + 15] = b3.w;

        barrier(CLK_LOCAL_MEM_FENCE);

        int my_m = ty * 8;
        int my_n = tx * 8;

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            float ra[8];
            float rb[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) ra[i] = lds_A[k][my_m + i];
            #pragma unroll
            for (int j = 0; j < 8; j++) rb[j] = lds_B[k][my_n + j];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[i][j] = mad(ra[i], rb[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write out
    int out_m = block_m + ty * 8;
    int out_n = block_n + tx * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j += 4) {
            float4 out_val = (float4)(acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]);
            vstore4(out_val, 0, &C[(out_m + i) * N + out_n + j]);
        }
    }
}
)CLC";

int main() {
    std::cout << "Testing Coalesced Vega 8 GEMM Performance...\n";
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, nullptr);

    size_t len = strlen(OPT_GEMM_SRC);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &OPT_GEMM_SRC, &len, nullptr);
    cl_int b_err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (b_err != CL_SUCCESS) {
        size_t sz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
        std::vector<char> log(sz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sz, log.data(), nullptr);
        std::cerr << log.data() << std::endl;
        return 1;
    }

    cl_kernel k = clCreateKernel(prog, "gemm_coalesced_wave64", nullptr);

    int M = 2048, N = 2048, K = 2048;
    size_t f_sz = (size_t)M * N * sizeof(float);
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, f_sz, nullptr, nullptr);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, f_sz, nullptr, nullptr);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, f_sz, nullptr, nullptr);

    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dA);
    clSetKernelArg(k, 4, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 5, sizeof(cl_mem), &dC);

    size_t g_sz[2] = { (size_t)N / 64 * 8, (size_t)M / 64 * 8 }; // 256, 256
    size_t l_sz[2] = { 8, 8 };

    // Warmup
    clEnqueueNDRangeKernel(queue, k, 2, nullptr, g_sz, l_sz, 0, nullptr, nullptr);
    clFinish(queue);

    // Profile 5 runs
    std::cout << "Benchmarking 2048x2048x2048 FP32 GEMM...\n";
    for (int r = 0; r < 5; r++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k, 2, nullptr, g_sz, l_sz, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);

        double ns = (double)(t1 - t0);
        double flops = 2.0 * (double)M * N * K;
        double tflops = (flops / ns) * 1e-3;
        std::cout << "  Run " << r << ": Time = " << std::fixed << std::setprecision(3) << (ns / 1e6) 
                  << " ms | TFLOPS = " << std::setprecision(3) << tflops << "\n";
    }

    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}
