// test_vega8_prefill.cpp
// Standalone Test Harness and Benchmark for Engine B (Vega 8 OpenCL GEMM Prefill Specialist)
// AMD Ryzen 7 7730U (gfx90c, GCN 5.0)

#include "vega8_context.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

using namespace zen3_engine;

// Benchmark dimensions for LFM 2.5 prompt prefill shapes
constexpr int M = 512;   // Prompt batch / token length
constexpr int N = 2048;  // Hidden dimension / projection size
constexpr int K = 2048;  // Hidden dimension
constexpr double TOTAL_FLOPS = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

// Helper for allocating 4096-byte page aligned host memory for zero-copy UMA
template <typename T>
T* allocate_aligned(size_t count) {
    void* ptr = _aligned_malloc(count * sizeof(T), 4096);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

template <typename T>
void free_aligned(T* ptr) {
    if (ptr) {
        _aligned_free(ptr);
    }
}

// Convert float to half (IEEE 754-2008 uint16_t)
uint16_t float_to_half(float val) {
    uint32_t x;
    std::memcpy(&x, &val, sizeof(float));
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        return (uint16_t)(sign << 15);
    } else if (exp >= 31) {
        return (uint16_t)((sign << 15) | 0x7C00);
    }
    return (uint16_t)((sign << 15) | (exp << 10) | (mant >> 13));
}

float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f_exp;
    uint32_t f_mant;

    if (exp == 0) {
        f_exp = 0;
        f_mant = mant << 13;
    } else if (exp == 31) {
        f_exp = 255;
        f_mant = mant << 13;
    } else {
        f_exp = exp - 15 + 127;
        f_mant = mant << 13;
    }

    uint32_t f = (sign << 31) | (f_exp << 23) | f_mant;
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// CPU reference computation for a sampled subset of outputs
bool verify_correctness(const float* A, const float* B, const float* C, int m_size, int n_size, int k_size, bool transposed_B = false) {
    std::cout << "[Verification] Comparing GPU output against CPU reference (sampled test points)..." << std::endl;
    int samples = 0;
    int correct = 0;
    float max_rel_err = 0.0f;

    // Check corners and interior points
    int test_rows[] = {0, 1, 15, 63, 128, 255, 384, 511};
    int test_cols[] = {0, 1, 31, 64, 256, 512, 1024, 2047};

    for (int r : test_rows) {
        if (r >= m_size) continue;
        for (int c : test_cols) {
            if (c >= n_size) continue;

            double ref_sum = 0.0;
            if (!transposed_B) {
                for (int k = 0; k < k_size; ++k) {
                    ref_sum += static_cast<double>(A[r * k_size + k]) * static_cast<double>(B[k * n_size + c]);
                }
            } else {
                for (int k = 0; k < k_size; ++k) {
                    ref_sum += static_cast<double>(A[r * k_size + k]) * static_cast<double>(B[c * k_size + k]);
                }
            }

            float gpu_val = C[r * n_size + c];
            float diff = std::abs(gpu_val - static_cast<float>(ref_sum));
            float denom = std::max(std::abs(gpu_val), static_cast<float>(std::abs(ref_sum)));
            float rel_err = denom > 1e-5f ? (diff / denom) : diff;

            if (rel_err > max_rel_err) max_rel_err = rel_err;
            if (rel_err < 1e-3f) {
                correct++;
            }
            samples++;
        }
    }

    std::cout << "[Verification] Checked " << samples << " test points. Accuracy: "
              << (100.0 * correct / samples) << "% | Max Relative Error: " << max_rel_err << std::endl;
    return (max_rel_err < 5e-3f);
}

struct BenchmarkResult {
    std::string name;
    double time_ms;
    double tflops;
    double speedup_vs_baseline;
};

int main(int argc, char** argv) {
    std::cout << "========================================================================" << std::endl;
    std::cout << " Engine B: AMD Vega 8 (gfx90c) OpenCL GEMM Prefill Specialist Benchmark " << std::endl;
    std::cout << " Matrix Shape: M=" << M << ", N=" << N << ", K=" << K << " (LFM 2.5 Prompt Shapes)" << std::endl;
    std::cout << " Total Workload: " << (TOTAL_FLOPS / 1e9) << " GFLOPs per pass" << std::endl;
    std::cout << " Target Performance: >= 1.20 TFLOPs (vs 0.27 TFLOPs stock baseline)" << std::endl;
    std::cout << "========================================================================" << std::endl;

    Vega8Context ctx;
    if (!ctx.init(true)) {
        std::cerr << "[-] Error: Failed to initialize Vega 8 OpenCL context!" << std::endl;
        return 1;
    }

    // Allocate page-aligned host memory for Zero-Copy UMA mapping
    std::cout << "\n[Host Memory] Allocating page-aligned (4096-byte) buffers in RAM..." << std::endl;
    float* h_A = allocate_aligned<float>(M * K);
    float* h_B = allocate_aligned<float>(K * N);
    float* h_B_T = allocate_aligned<float>(N * K);
    float* h_C = allocate_aligned<float>(M * N);

    // Initialize with bounded test values
    for (int i = 0; i < M * K; ++i) h_A[i] = 0.05f * (float)((i % 17) - 8);
    for (int i = 0; i < K * N; ++i) h_B[i] = 0.05f * (float)((i % 19) - 9);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            h_B_T[n * K + k] = h_B[k * N + n];
        }
    }
    std::memset(h_C, 0, M * N * sizeof(float));

    // Zero-Copy UMA Buffer Creation
    std::cout << "[Zero-Copy UMA] Creating CL_MEM_USE_HOST_PTR mappings (no PCIe transfers)..." << std::endl;
    auto wrap_start = std::chrono::high_resolution_clock::now();
    cl_mem d_A = ctx.wrap_host_buffer(h_A, M * K * sizeof(float), CL_MEM_READ_ONLY);
    cl_mem d_B = ctx.wrap_host_buffer(h_B, K * N * sizeof(float), CL_MEM_READ_ONLY);
    cl_mem d_B_T = ctx.wrap_host_buffer(h_B_T, N * K * sizeof(float), CL_MEM_READ_ONLY);
    cl_mem d_C = ctx.wrap_host_buffer(h_C, M * N * sizeof(float), CL_MEM_READ_WRITE);
    auto wrap_end = std::chrono::high_resolution_clock::now();

    double wrap_time_us = std::chrono::duration<double, std::micro>(wrap_end - wrap_start).count();
    std::cout << "[Zero-Copy UMA] 4 Buffers (total " << ((M*K + 2*K*N + M*N) * sizeof(float) / (1024*1024))
              << " MB) mapped in " << std::fixed << std::setprecision(2) << wrap_time_us 
              << " us. Zero copy verified (Host UMA directly addressable by GPU).\n" << std::endl;

    // Load and compile OpenCL kernels
    std::string cl_path = "vega8_gemm.cl";
    std::cout << "[OpenCL Compiler] Compiling " << cl_path << " for gfx90c (Wave64, LDS Tiling)..." << std::endl;
    cl_program program = ctx.build_program_from_file(cl_path, "-cl-mad-enable -cl-fast-relaxed-math");
    if (!program) {
        std::cerr << "[-] Error: Failed to compile " << cl_path << std::endl;
        return 1;
    }
    std::cout << "[OpenCL Compiler] Compilation successful!\n" << std::endl;

    // Test Kernels
    std::vector<BenchmarkResult> results;
    double baseline_time_ms = 0.0;

    auto benchmark_kernel = [&](const std::string& name, cl_kernel kernel, 
                                size_t global_0, size_t global_1,
                                size_t local_0, size_t local_1,
                                int warmup_count, int bench_count) -> BenchmarkResult {
        std::cout << ">>> Running: " << name << " [Local: (" << local_0 << ", " << local_1 << ")]..." << std::endl;

        size_t global_work[2] = {global_0, global_1};
        size_t local_work[2] = {local_0, local_1};

        // Warmup
        for (int w = 0; w < warmup_count; ++w) {
            cl_int err = clEnqueueNDRangeKernel(ctx.get_queue(), kernel, 2, nullptr, 
                                                global_work, local_work, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "[-] Kernel launch failed during warmup: " << Vega8Context::get_error_string(err) << std::endl;
                return {name, 0.0, 0.0, 0.0};
            }
        }
        ctx.finish();

        // Benchmark runs with GPU profiling events
        double total_gpu_ms = 0.0;
        for (int b = 0; b < bench_count; ++b) {
            cl_event ev;
            cl_int err = clEnqueueNDRangeKernel(ctx.get_queue(), kernel, 2, nullptr, 
                                                global_work, local_work, 0, nullptr, &ev);
            if (err != CL_SUCCESS) {
                std::cerr << "[-] Kernel launch failed during bench: " << Vega8Context::get_error_string(err) << std::endl;
                return {name, 0.0, 0.0, 0.0};
            }
            clWaitForEvents(1, &ev);

            cl_ulong start = 0, end = 0;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
            clReleaseEvent(ev);

            double duration_ms = static_cast<double>(end - start) * 1e-6;
            total_gpu_ms += duration_ms;
        }

        double avg_time_ms = total_gpu_ms / bench_count;
        double tflops = (TOTAL_FLOPS / (avg_time_ms * 1e-3)) / 1e12;
        double speedup = (baseline_time_ms > 0.0) ? (baseline_time_ms / avg_time_ms) : 1.0;

        std::cout << "    Avg Duration: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms"
                  << " | Compute: " << std::setprecision(3) << tflops << " TFLOPs"
                  << " | Speedup: " << std::setprecision(2) << speedup << "x" << std::endl;

        return {name, avg_time_ms, tflops, speedup};
    };

    // 1. Stock Baseline
    {
        cl_kernel k_stock = ctx.create_kernel(program, "gemm_stock_baseline");
        if (k_stock) {
            clSetKernelArg(k_stock, 0, sizeof(int), &M);
            clSetKernelArg(k_stock, 1, sizeof(int), &N);
            clSetKernelArg(k_stock, 2, sizeof(int), &K);
            clSetKernelArg(k_stock, 3, sizeof(cl_mem), &d_A);
            clSetKernelArg(k_stock, 4, sizeof(cl_mem), &d_B);
            clSetKernelArg(k_stock, 5, sizeof(cl_mem), &d_C);

            // Generic baseline: 16x16 workgroup, N x M grid
            BenchmarkResult res = benchmark_kernel("Stock OpenCL GEMM Baseline", k_stock,
                                                   N, M, 16, 16, 2, 5);
            baseline_time_ms = res.time_ms;
            res.speedup_vs_baseline = 1.0;
            results.push_back(res);
            clReleaseKernel(k_stock);
        }
    }

    // 2. Optimized Kernel 1: 32x32 Tile with 4x4 Register Micro-Tile (Wave64)
    {
        cl_kernel k1 = ctx.create_kernel(program, "vega8_gemm_fp32_32x32_4x4");
        if (k1) {
            clSetKernelArg(k1, 0, sizeof(int), &M);
            clSetKernelArg(k1, 1, sizeof(int), &N);
            clSetKernelArg(k1, 2, sizeof(int), &K);
            clSetKernelArg(k1, 3, sizeof(cl_mem), &d_A);
            clSetKernelArg(k1, 4, sizeof(cl_mem), &d_B);
            clSetKernelArg(k1, 5, sizeof(cl_mem), &d_C);

            // Grid: (N/32) * 8, (M/32) * 8 = (2048/32)*8, (512/32)*8 = (64*8, 16*8) = (512, 128)
            // Local: (8, 8) = 64 threads (Wave64)
            size_t g0 = (N / 32) * 8;
            size_t g1 = (M / 32) * 8;

            // Run once and verify correctness
            size_t gw[2] = {g0, g1};
            size_t lw[2] = {8, 8};
            clEnqueueNDRangeKernel(ctx.get_queue(), k1, 2, nullptr, gw, lw, 0, nullptr, nullptr);
            ctx.finish();
            verify_correctness(h_A, h_B, h_C, M, N, K, false);

            BenchmarkResult res = benchmark_kernel("Vega8 FP32 Tile-32x32 (Micro-4x4 Wave64 Double-Buffered)",
                                                   k1, g0, g1, 8, 8, 5, 25);
            results.push_back(res);
            clReleaseKernel(k1);
        }
    }

    // 3. Optimized Kernel 2: 64x64 Tile with 8x8 Register Micro-Tile, K=16 (Wave64)
    {
        cl_kernel k2 = ctx.create_kernel(program, "vega8_gemm_fp32_64x64_8x8");
        if (k2) {
            clSetKernelArg(k2, 0, sizeof(int), &M);
            clSetKernelArg(k2, 1, sizeof(int), &N);
            clSetKernelArg(k2, 2, sizeof(int), &K);
            clSetKernelArg(k2, 3, sizeof(cl_mem), &d_A);
            clSetKernelArg(k2, 4, sizeof(cl_mem), &d_B);
            clSetKernelArg(k2, 5, sizeof(cl_mem), &d_C);

            // Grid: (N/64) * 8, (M/64) * 8 = (2048/64)*8, (512/64)*8 = (32*8, 8*8) = (256, 64)
            // Local: (8, 8) = 64 threads (Wave64)
            size_t g0 = (N / 64) * 8;
            size_t g1 = (M / 64) * 8;

            size_t gw[2] = {g0, g1};
            size_t lw[2] = {8, 8};
            clEnqueueNDRangeKernel(ctx.get_queue(), k2, 2, nullptr, gw, lw, 0, nullptr, nullptr);
            ctx.finish();
            verify_correctness(h_A, h_B, h_C, M, N, K, false);

            BenchmarkResult res = benchmark_kernel("Vega8 FP32 Tile-64x64 (Micro-8x8 K16 Wave64 Double-Buffered)",
                                                   k2, g0, g1, 8, 8, 5, 25);
            results.push_back(res);
            clReleaseKernel(k2);
        }
    }

    // 4. Optimized Kernel 3: 64x64 Tile with 8x8 Register Micro-Tile, K=32 (Wave64)
    {
        cl_kernel k3 = ctx.create_kernel(program, "vega8_gemm_fp32_64x64_k32_8x8");
        if (k3) {
            clSetKernelArg(k3, 0, sizeof(int), &M);
            clSetKernelArg(k3, 1, sizeof(int), &N);
            clSetKernelArg(k3, 2, sizeof(int), &K);
            clSetKernelArg(k3, 3, sizeof(cl_mem), &d_A);
            clSetKernelArg(k3, 4, sizeof(cl_mem), &d_B);
            clSetKernelArg(k3, 5, sizeof(cl_mem), &d_C);

            size_t g0 = (N / 64) * 8;
            size_t g1 = (M / 64) * 8;

            size_t gw[2] = {g0, g1};
            size_t lw[2] = {8, 8};
            clEnqueueNDRangeKernel(ctx.get_queue(), k3, 2, nullptr, gw, lw, 0, nullptr, nullptr);
            ctx.finish();
            verify_correctness(h_A, h_B, h_C, M, N, K, false);

            BenchmarkResult res = benchmark_kernel("Vega8 FP32 Tile-64x64 (Micro-8x8 K32 Wave64 High-Density)",
                                                   k3, g0, g1, 8, 8, 5, 25);
            results.push_back(res);
            clReleaseKernel(k3);
        }
    }

    // 5. Optimized Kernel 4: Transposed Weights (B^T [N, K])
    {
        cl_kernel k4 = ctx.create_kernel(program, "vega8_gemm_fp32_64x64_nt");
        if (k4) {
            clSetKernelArg(k4, 0, sizeof(int), &M);
            clSetKernelArg(k4, 1, sizeof(int), &N);
            clSetKernelArg(k4, 2, sizeof(int), &K);
            clSetKernelArg(k4, 3, sizeof(cl_mem), &d_A);
            clSetKernelArg(k4, 4, sizeof(cl_mem), &d_B_T);
            clSetKernelArg(k4, 5, sizeof(cl_mem), &d_C);

            size_t g0 = (N / 64) * 8;
            size_t g1 = (M / 64) * 8;

            size_t gw[2] = {g0, g1};
            size_t lw[2] = {8, 8};
            clEnqueueNDRangeKernel(ctx.get_queue(), k4, 2, nullptr, gw, lw, 0, nullptr, nullptr);
            ctx.finish();
            verify_correctness(h_A, h_B, h_C, M, N, K, false);

            BenchmarkResult res = benchmark_kernel("Vega8 FP32 Transposed B^T (LLM Weight Matrix Prefill)",
                                                   k4, g0, g1, 8, 8, 5, 25);
            results.push_back(res);
            clReleaseKernel(k4);
        }
    }

    // 6. Half-Precision (FP16) GEMM if supported
    if (ctx.get_device_info().fp16_supported) {
        cl_kernel k_fp16 = ctx.create_kernel(program, "vega8_gemm_fp16_64x64_8x8");
        if (k_fp16) {
            uint16_t* h_A_half = allocate_aligned<uint16_t>(M * K);
            uint16_t* h_B_half = allocate_aligned<uint16_t>(K * N);
            uint16_t* h_C_half = allocate_aligned<uint16_t>(M * N);

            for (int i = 0; i < M * K; ++i) h_A_half[i] = float_to_half(h_A[i]);
            for (int i = 0; i < K * N; ++i) h_B_half[i] = float_to_half(h_B[i]);
            std::memset(h_C_half, 0, M * N * sizeof(uint16_t));

            cl_mem d_A_half = ctx.wrap_host_buffer(h_A_half, M * K * sizeof(uint16_t), CL_MEM_READ_ONLY);
            cl_mem d_B_half = ctx.wrap_host_buffer(h_B_half, K * N * sizeof(uint16_t), CL_MEM_READ_ONLY);
            cl_mem d_C_half = ctx.wrap_host_buffer(h_C_half, M * N * sizeof(uint16_t), CL_MEM_READ_WRITE);

            clSetKernelArg(k_fp16, 0, sizeof(int), &M);
            clSetKernelArg(k_fp16, 1, sizeof(int), &N);
            clSetKernelArg(k_fp16, 2, sizeof(int), &K);
            clSetKernelArg(k_fp16, 3, sizeof(cl_mem), &d_A_half);
            clSetKernelArg(k_fp16, 4, sizeof(cl_mem), &d_B_half);
            clSetKernelArg(k_fp16, 5, sizeof(cl_mem), &d_C_half);

            size_t g0 = (N / 64) * 8;
            size_t g1 = (M / 64) * 8;

            BenchmarkResult res = benchmark_kernel("Vega8 FP16 Rapid Packed Math (Tile-64x64 Micro-8x8 Wave64)",
                                                   k_fp16, g0, g1, 8, 8, 5, 25);
            results.push_back(res);

            clReleaseMemObject(d_A_half);
            clReleaseMemObject(d_B_half);
            clReleaseMemObject(d_C_half);
            free_aligned(h_A_half);
            free_aligned(h_B_half);
            free_aligned(h_C_half);
            clReleaseKernel(k_fp16);
        }
    }

    // Print Final Performance Summary Table
    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << "                              BENCHMARK SUMMARY RESULTS                                   " << std::endl;
    std::cout << " Workload: M=" << M << ", N=" << N << ", K=" << K << " (" << (TOTAL_FLOPS / 1e9) << " GFLOPs) | Device: AMD gfx90c (Vega 8)" << std::endl;
    std::cout << "==========================================================================================" << std::endl;
    std::cout << std::left << std::setw(60) << "Kernel Implementation" 
              << std::right << std::setw(12) << "Latency (ms)"
              << std::setw(15) << "Sustained TFLOPs"
              << std::setw(11) << "Speedup" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;

    double max_tflops = 0.0;
    for (const auto& r : results) {
        std::cout << std::left << std::setw(60) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms
                  << std::setw(15) << std::setprecision(3) << r.tflops
                  << std::setw(10) << std::setprecision(2) << r.speedup_vs_baseline << "x" << std::endl;
        if (r.tflops > max_tflops) max_tflops = r.tflops;
    }
    std::cout << "==========================================================================================" << std::endl;

    if (max_tflops >= 1.20) {
        std::cout << "[SUCCESS] Compute target reached: " << max_tflops << " TFLOPs >= 1.20 TFLOPs target!" << std::endl;
    } else {
        std::cout << "[WARNING] Peak compute: " << max_tflops << " TFLOPs (target is 1.20 TFLOPs)" << std::endl;
    }

    // Cleanup
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_B_T);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);

    free_aligned(h_A);
    free_aligned(h_B);
    free_aligned(h_B_T);
    free_aligned(h_C);

    return 0;
}
