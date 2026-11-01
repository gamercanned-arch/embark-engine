#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

const char* TRANSPOSED_RPM_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 2

// -------------------------------------------------------------------------
// Transposed LDS Rapid Packed Math GEMM on AMD Vega 8 (gfx90c)
// 1. Matrix A is transposed during LDS store: lds_AT[k][m]
//    - Zero conflicts on store: threads 0..63 write contiguous elements of row k!
//    - Zero conflicts on read: ty loads 8 contiguous rows via two half4 ds_read_b64
//    - 8 threads along tx share the exact same ty -> 100% single-cycle LDS broadcast!
// 2. Matrix B is stored as lds_B[k][col]:
//    - tx loads 8 contiguous columns via two half4 ds_read_b64
//    - 8 threads along ty share the exact same tx -> 100% single-cycle LDS broadcast!
// 3. Exactly 4 vector LDS reads per k-step (instead of 12 scalar reads)!
// 4. Inner loop executes 32 v_pk_fma_f16 instructions (64 FLOPs) entirely in registers!
// -------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_rpm_transposed(
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

    // Transposed LDS for A: [32 k-steps][64 rows + 2 pad]
    __local half lds_AT[32][64 + LDS_PAD];
    // LDS for B: [32 k-steps][64 cols + 2 pad]
    __local half lds_B[32][64 + LDS_PAD];

    half2 c2[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c2[i][j] = (half2)(0.0h, 0.0h);
    }

    const int load_row_a = tid; // 0..63
    const int global_row_a = block_m + load_row_a;

    const int load_row_b = tid >> 1;        // 0..31
    const int load_col_b = (tid & 1) << 5;   // 0 or 32
    const int global_col_b = block_n + load_col_b;

    const int num_tiles = K / 32;

    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 32;

        // Load 32 halfs of A coalesced from global memory
        half16 a_raw0 = vload16(0, &A[global_row_a * K + k_base + 0]);
        half16 a_raw1 = vload16(0, &A[global_row_a * K + k_base + 16]);

        // Transpose A into lds_AT: each thread writes its 32 elements across k=0..31
        // Across the 64 threads, threads 0..63 write contiguous elements of column load_row_a!
        #pragma unroll
        for (int ki = 0; ki < 16; ki++) {
            lds_AT[ki + 0][load_row_a]  = ((half*)&a_raw0)[ki];
            lds_AT[ki + 16][load_row_a] = ((half*)&a_raw1)[ki];
        }

        // Load 32 halfs of B coalesced from global memory
        half16 b_raw0 = vload16(0, &B[(k_base + load_row_b) * N + global_col_b + 0]);
        half16 b_raw1 = vload16(0, &B[(k_base + load_row_b) * N + global_col_b + 16]);

        vstore16(b_raw0, 0, &lds_B[load_row_b][load_col_b + 0]);
        vstore16(b_raw1, 0, &lds_B[load_row_b][load_col_b + 16]);

        barrier(CLK_LOCAL_MEM_FENCE);

        const int my_m = ty * 8; // 8 rows for this thread
        const int my_n = tx * 8; // 8 columns for this thread

        // Inner compute loop with 64-bit Vector LDS reads
        #pragma unroll 16
        for (int k = 0; k < 32; k++) {
            // Load 8 contiguous rows of A as two half4s (two 64-bit vector loads!)
            half4 a_vec0 = vload4(0, &lds_AT[k][my_m + 0]);
            half4 a_vec1 = vload4(0, &lds_AT[k][my_m + 4]);

            // Load 8 contiguous columns of B as two half4s (two 64-bit vector loads!)
            half4 b_vec0 = vload4(0, &lds_B[k][my_n + 0]);
            half4 b_vec1 = vload4(0, &lds_B[k][my_n + 4]);

            // Pack B elements into half2
            half2 b_pack[4];
            b_pack[0] = (half2)(b_vec0.s0, b_vec0.s1);
            b_pack[1] = (half2)(b_vec0.s2, b_vec0.s3);
            b_pack[2] = (half2)(b_vec1.s0, b_vec1.s1);
            b_pack[3] = (half2)(b_vec1.s2, b_vec1.s3);

            // Rapid Packed Math: 32 v_pk_fma_f16 instructions
            half2 a_pk0 = (half2)(a_vec0.s0, a_vec0.s0);
            c2[0][0] = fma(a_pk0, b_pack[0], c2[0][0]);
            c2[0][1] = fma(a_pk0, b_pack[1], c2[0][1]);
            c2[0][2] = fma(a_pk0, b_pack[2], c2[0][2]);
            c2[0][3] = fma(a_pk0, b_pack[3], c2[0][3]);

            half2 a_pk1 = (half2)(a_vec0.s1, a_vec0.s1);
            c2[1][0] = fma(a_pk1, b_pack[0], c2[1][0]);
            c2[1][1] = fma(a_pk1, b_pack[1], c2[1][1]);
            c2[1][2] = fma(a_pk1, b_pack[2], c2[1][2]);
            c2[1][3] = fma(a_pk1, b_pack[3], c2[1][3]);

            half2 a_pk2 = (half2)(a_vec0.s2, a_vec0.s2);
            c2[2][0] = fma(a_pk2, b_pack[0], c2[2][0]);
            c2[2][1] = fma(a_pk2, b_pack[1], c2[2][1]);
            c2[2][2] = fma(a_pk2, b_pack[2], c2[2][2]);
            c2[2][3] = fma(a_pk2, b_pack[3], c2[2][3]);

            half2 a_pk3 = (half2)(a_vec0.s3, a_vec0.s3);
            c2[3][0] = fma(a_pk3, b_pack[0], c2[3][0]);
            c2[3][1] = fma(a_pk3, b_pack[1], c2[3][1]);
            c2[3][2] = fma(a_pk3, b_pack[2], c2[3][2]);
            c2[3][3] = fma(a_pk3, b_pack[3], c2[3][3]);

            half2 a_pk4 = (half2)(a_vec1.s0, a_vec1.s0);
            c2[4][0] = fma(a_pk4, b_pack[0], c2[4][0]);
            c2[4][1] = fma(a_pk4, b_pack[1], c2[4][1]);
            c2[4][2] = fma(a_pk4, b_pack[2], c2[4][2]);
            c2[4][3] = fma(a_pk4, b_pack[3], c2[4][3]);

            half2 a_pk5 = (half2)(a_vec1.s1, a_vec1.s1);
            c2[5][0] = fma(a_pk5, b_pack[0], c2[5][0]);
            c2[5][1] = fma(a_pk5, b_pack[1], c2[5][1]);
            c2[5][2] = fma(a_pk5, b_pack[2], c2[5][2]);
            c2[5][3] = fma(a_pk5, b_pack[3], c2[5][3]);

            half2 a_pk6 = (half2)(a_vec1.s2, a_vec1.s2);
            c2[6][0] = fma(a_pk6, b_pack[0], c2[6][0]);
            c2[6][1] = fma(a_pk6, b_pack[1], c2[6][1]);
            c2[6][2] = fma(a_pk6, b_pack[2], c2[6][2]);
            c2[6][3] = fma(a_pk6, b_pack[3], c2[6][3]);

            half2 a_pk7 = (half2)(a_vec1.s3, a_vec1.s3);
            c2[7][0] = fma(a_pk7, b_pack[0], c2[7][0]);
            c2[7][1] = fma(a_pk7, b_pack[1], c2[7][1]);
            c2[7][2] = fma(a_pk7, b_pack[2], c2[7][2]);
            c2[7][3] = fma(a_pk7, b_pack[3], c2[7][3]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write back 8x8 results coalesced
    #pragma unroll
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

    size_t local_work_size[3] = {8, 8, 1};
    size_t global_work_size[3] = {
        ((size_t)N + 63) / 64 * 8,
        ((size_t)M + 63) / 64 * 8,
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
    std::cout << "Transposed Vector LDS Rapid Packed Math Benchmark on AMD Vega 8" << std::endl;
    std::cout << "====================================================================" << std::endl;

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, NULL, &err);

    cl_program prog = clCreateProgramWithSource(ctx, 1, &TRANSPOSED_RPM_SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "OpenCL Build Error:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(prog, "gemm_rpm_transposed", &err);

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
