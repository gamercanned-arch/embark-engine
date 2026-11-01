// test_llama_gemm.cpp
#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <malloc.h>

using namespace zen3_engine;

constexpr int M = 512;
constexpr int N = 2048;
constexpr int K = 2048;
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

int main() {
    Vega8Context ctx;
    ctx.init();

    uint16_t* h_A = (uint16_t*)_aligned_malloc(M * K * sizeof(uint16_t), 4096);
    float* h_B = (float*)_aligned_malloc(K * N * sizeof(float), 4096);
    float* h_C = (float*)_aligned_malloc(M * N * sizeof(float), 4096);

    for (int i = 0; i < M * K; ++i) h_A[i] = 0x3C00;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

    cl_mem d_A = ctx.wrap_host_buffer(h_A, M * K * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_B = ctx.wrap_host_buffer(h_B, K * N * sizeof(float), CL_MEM_READ_ONLY);
    cl_mem d_C = ctx.wrap_host_buffer(h_C, M * N * sizeof(float), CL_MEM_READ_WRITE);

    cl_program prog = ctx.build_program_from_file("c:\\Users\\abhik\\Documents\\antigravity\\silly-pascal\\llama.cpp\\ggml\\src\\ggml-opencl\\kernels\\mul_mat_f16_f32.cl");
    if (!prog) return 1;

    cl_kernel k = ctx.create_kernel(prog, "mul_mat_f16_f32");
    if (!k) return 1;

    cl_ulong offsetA = 0, offsetB = 0, offsetC = 0;
    clSetKernelArg(k, 0, sizeof(int), &M);
    clSetKernelArg(k, 1, sizeof(int), &N);
    clSetKernelArg(k, 2, sizeof(int), &K);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(k, 4, sizeof(cl_ulong), &offsetA);
    clSetKernelArg(k, 5, sizeof(cl_mem), &d_B);
    clSetKernelArg(k, 6, sizeof(cl_ulong), &offsetB);
    clSetKernelArg(k, 7, sizeof(cl_mem), &d_C);
    clSetKernelArg(k, 8, sizeof(cl_ulong), &offsetC);

    // In mul_mat_f16_f32:
    // offsetM = get_group_id(0) * OPWM (64)
    // offsetN = get_group_id(1) * OPWN (64)
    // Global work size: (M / 64) * WG_M, (N / 64) * WG_N = (512/64)*16 = 128, (2048/64)*8 = 256
    // Local work size: (WG_M, WG_N) = (16, 8)
    size_t gw[2] = {128, 256};
    size_t lw[2] = {16, 8};

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
        clEnqueueNDRangeKernel(ctx.get_queue(), k, 2, nullptr, gw, lw, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong s, e;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr);
        clReleaseEvent(ev);
        total_ms += (e - s) * 1e-6;
    }

    double avg_ms = total_ms / iters;
    double tflops = (TOTAL_FLOPS / (avg_ms * 1e-3)) / 1e12;
    std::cout << "[llama.cpp mul_mat_f16_f32] Avg: " << avg_ms << " ms | TFLOPs: " << tflops << std::endl;

    return 0;
}
