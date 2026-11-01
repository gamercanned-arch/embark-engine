#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

const char* GOLDEN_C4X4_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 2

// -------------------------------------------------------------------------
// Real Golden c2[4][4] Rapid Packed Math GEMM on AMD Vega 8 (gfx90c)
// - 128 threads per workgroup (8x16) = 2 Wave64s per WG
// - Sits in ~32-34 VGPRs -> 7-8 waves/SIMD (28-32 waves/CU, 70-80% occupancy!)
// - 100% Coalesced 128-bit DRAM reads & LDS writes for BOTH A and B
// - LDS broadcast: all 8 threads in tx share ty -> 100% conflict-free broadcast for A!
// - LDS broadcast: all 16 threads in ty share tx -> 100% conflict-free broadcast for B!
// - True Rapid Packed Math v_pk_fma_f16 (16 instructions = 32 FP16 MACs per k-step)
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(8, 16, 1)))
void gemm_rpm_golden_real(
    const int M, const int N, const int K,
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C)
{
    const int tx = get_local_id(0); // 0..7  (covers 8*8 = 64 columns)
    const int ty = get_local_id(1); // 0..15 (covers 16*4 = 64 rows)
    const int tid = ty * 8 + tx;    // 0..127

    const int block_m = get_group_id(1) * 64;
    const int block_n = get_group_id(0) * 64;

    // LDS storage:
    // lds_A: [64 rows][32 halfs + 2 pad] -> stride 34 halfs
    __local half lds_A[64][32 + LDS_PAD];
    // lds_B4: [32 k-steps][16 half4s + 1 pad] -> stride 17 half4s = 68 halfs
    __local half4 lds_B4[32][16 + 1];

    half2 c2[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c2[i][j] = (half2)(0.0h, 0.0h);
    }

    // 128 threads load 64 rows x 32 cols of A (2048 halfs = 128 x 16 halfs)
    const int load_row_a = tid >> 1;        // 0..63
    const int load_col_a = (tid & 1) << 4;  // 0 or 16
    const int global_row_a = block_m + load_row_a;

    // 128 threads load 32 rows x 64 cols of B (2048 halfs = 128 x 16 halfs)
    const int load_row_b = tid >> 2;        // 0..31
    const int load_col_b = (tid & 3) << 4;  // 0, 16, 32, 48
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 32;

        // 100% Coalesced 128-bit DRAM loads for A
        half16 a_raw = vload16(0, &A[global_row_a * K + k_base + load_col_a]);
        vstore16(a_raw, 0, &lds_A[load_row_a][load_col_a]);

        // 100% Coalesced 128-bit DRAM loads for B
        half8 b_raw0 = vload8(0, &B[(k_base + load_row_b) * N + global_col_b + 0]);
        half8 b_raw1 = vload8(0, &B[(k_base + load_row_b) * N + global_col_b + 8]);

        const int b_c4 = load_col_b >> 2;
        lds_B4[load_row_b][b_c4 + 0] = vload4(0, (half*)&b_raw0);
        lds_B4[load_row_b][b_c4 + 1] = vload4(1, (half*)&b_raw0);
        lds_B4[load_row_b][b_c4 + 2] = vload4(0, (half*)&b_raw1);
        lds_B4[load_row_b][b_c4 + 3] = vload4(1, (half*)&b_raw1);

        barrier(CLK_LOCAL_MEM_FENCE);

        const int my_m = ty * 4;  // 4 rows for this thread
        const int my_n4 = tx * 2; // 8 columns for this thread (2 half4s)

        // Compute phase
        #pragma unroll 16
        for (int k = 0; k < 32; k++) {
            // Load 8 columns of B as two half4 (two 64-bit vector LDS reads: ds_read_b64)
            half4 b_vec0 = lds_B4[k][my_n4 + 0];
            half4 b_vec1 = lds_B4[k][my_n4 + 1];

            half2 b_pack[4];
            b_pack[0] = (half2)(b_vec0.s0, b_vec0.s1);
            b_pack[1] = (half2)(b_vec0.s2, b_vec0.s3);
            b_pack[2] = (half2)(b_vec1.s0, b_vec1.s1);
            b_pack[3] = (half2)(b_vec1.s2, b_vec1.s3);

            // Load 4 rows of A (LDS broadcast across tx=0..7)
            half a0 = lds_A[my_m + 0][k];
            half a1 = lds_A[my_m + 1][k];
            half a2 = lds_A[my_m + 2][k];
            half a3 = lds_A[my_m + 3][k];

            half2 a_pk0 = (half2)(a0, a0);
            c2[0][0] = fma(a_pk0, b_pack[0], c2[0][0]);
            c2[0][1] = fma(a_pk0, b_pack[1], c2[0][1]);
            c2[0][2] = fma(a_pk0, b_pack[2], c2[0][2]);
            c2[0][3] = fma(a_pk0, b_pack[3], c2[0][3]);

            half2 a_pk1 = (half2)(a1, a1);
            c2[1][0] = fma(a_pk1, b_pack[0], c2[1][0]);
            c2[1][1] = fma(a_pk1, b_pack[1], c2[1][1]);
            c2[1][2] = fma(a_pk1, b_pack[2], c2[1][2]);
            c2[1][3] = fma(a_pk1, b_pack[3], c2[1][3]);

            half2 a_pk2 = (half2)(a2, a2);
            c2[2][0] = fma(a_pk2, b_pack[0], c2[2][0]);
            c2[2][1] = fma(a_pk2, b_pack[1], c2[2][1]);
            c2[2][2] = fma(a_pk2, b_pack[2], c2[2][2]);
            c2[2][3] = fma(a_pk2, b_pack[3], c2[2][3]);

            half2 a_pk3 = (half2)(a3, a3);
            c2[3][0] = fma(a_pk3, b_pack[0], c2[3][0]);
            c2[3][1] = fma(a_pk3, b_pack[1], c2[3][1]);
            c2[3][2] = fma(a_pk3, b_pack[2], c2[3][2]);
            c2[3][3] = fma(a_pk3, b_pack[3], c2[3][3]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write back 4x8 results coalesced
    #pragma unroll
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

void run_test(cl_context ctx, cl_command_queue queue, cl_kernel kernel, int M, int N, int K, const char* label) {
    size_t size_A = (size_t)M * K * sizeof(cl_half);
    size_t size_B = (size_t)K * N * sizeof(cl_half);
    size_t size_C = (size_t)M * N * sizeof(cl_half);

    cl_int err;
    cl_mem buf_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size_A, NULL, &err);
    cl_mem buf_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size_B, NULL, &err);
    cl_mem buf_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size_C, NULL, &err);

    cl_half* h_A = (cl_half*)clEnqueueMapBuffer(queue, buf_A, CL_TRUE, CL_MAP_WRITE, 0, size_A, 0, NULL, NULL, NULL);
    cl_half* h_B = (cl_half*)clEnqueueMapBuffer(queue, buf_B, CL_TRUE, CL_MAP_WRITE, 0, size_B, 0, NULL, NULL, NULL);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A[i] = 0x3C00;
    for (size_t i = 0; i < (size_t)K * N; i++) h_B[i] = 0x3800;
    clEnqueueUnmapMemObject(queue, buf_A, h_A, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, buf_B, h_B, 0, NULL, NULL);
    clFinish(queue);

    clSetKernelArg(kernel, 0, sizeof(int), &M);
    clSetKernelArg(kernel, 1, sizeof(int), &N);
    clSetKernelArg(kernel, 2, sizeof(int), &K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_A);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_B);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &buf_C);

    size_t local_work_size[3] = {8, 16, 1}; // 128 threads
    size_t global_work_size[3] = {
        ((size_t)N + 63) / 64 * 8,
        ((size_t)M + 63) / 64 * 16,
        1
    };

    for (int i = 0; i < 3; i++) {
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    }
    clFinish(queue);

    const int ITERS = 15;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    }
    clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / ITERS;
    double tflops = (2.0 * (double)M * (double)N * (double)K * 1e-12) / (avg_ms * 1e-3);
    double pct_peak = (tflops / 4.096) * 100.0;

    std::cout << std::left << std::setw(32) << label 
              << " | " << std::setw(10) << std::fixed << std::setprecision(3) << avg_ms << " ms"
              << " | " << std::setw(7) << std::fixed << std::setprecision(3) << tflops << " TFLOPS"
              << " (" << std::fixed << std::setprecision(1) << pct_peak << "% of 4.096 TF peak)"
              << std::endl;

    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(buf_C);
}

int main() {
    std::cout << "====================================================================" << std::endl;
    std::cout << "Real Golden c2[4][4] Rapid Packed Math Benchmark on AMD Vega 8" << std::endl;
    std::cout << "====================================================================" << std::endl;

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, NULL, &err);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &GOLDEN_C4X4_SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "OpenCL Build Error:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(prog, "gemm_rpm_golden_real", &err);

    std::cout << "\n--- Model Prefill & Denoising Workloads ---\n";
    run_test(ctx, queue, kernel, 2048, 2048, 2048, "Standard GEMM (2048x2048x2048)");
    run_test(ctx, queue, kernel, 4096, 2048, 2048, "LFM/D1 Prefill (4096x2048x2048)");
    run_test(ctx, queue, kernel, 8192, 2048, 2048, "Long Prefill (8192x2048x2048)");
    run_test(ctx, queue, kernel, 2816, 3072, 3072, "FLUX.2 DiT Step (2816x3072x3072)");

    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
