#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <thread>
#include <atomic>
#include <immintrin.h>

int main() {
    std::ifstream f("src/d1_engine/opencl/vega8_d1_gemm.cl");
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, nullptr);

    const char* c_src = src.c_str();
    size_t sz = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &c_src, &sz, nullptr);
    cl_int b_err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (b_err != CL_SUCCESS) return 1;

    cl_kernel k_fp16 = clCreateKernel(prog, "vega8_d1_gemm_fp16_64x64_8x8", nullptr);

    int M = 2048, N = 2048, K = 2048;
    size_t bytes_fp16 = (size_t)M * K * sizeof(uint16_t);
    size_t g_ws[2] = { (size_t)N / 8, (size_t)M / 8 };
    size_t l_ws[2] = { 8, 8 };

    // =========================================================================
    // Test 1: Device Memory (Dedicated GPU Buffer)
    // =========================================================================
    std::cout << "[Experiment 1: Device VRAM Buffers (CPU Idle)]\n";
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes_fp16, nullptr, nullptr);
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes_fp16, nullptr, nullptr);
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_fp16, nullptr, nullptr);

    clSetKernelArg(k_fp16, 0, sizeof(int), &M);
    clSetKernelArg(k_fp16, 1, sizeof(int), &N);
    clSetKernelArg(k_fp16, 2, sizeof(int), &K);
    clSetKernelArg(k_fp16, 3, sizeof(cl_mem), &d_A);
    clSetKernelArg(k_fp16, 4, sizeof(cl_mem), &d_B);
    clSetKernelArg(k_fp16, 5, sizeof(cl_mem), &d_C);

    clEnqueueNDRangeKernel(queue, k_fp16, 2, nullptr, g_ws, l_ws, 0, nullptr, nullptr);
    clFinish(queue);

    double dev_ms = 0;
    for (int i = 0; i < 3; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_fp16, 2, nullptr, g_ws, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        dev_ms = (double)(t1 - t0) * 1e-6;
    }
    std::cout << "  Latency: " << dev_ms << " ms | TFLOPS: " << (2.0 * M * N * K / (dev_ms * 1e-3)) * 1e-12 << "\n\n";

    // =========================================================================
    // Test 2: Zero-Copy UMA Buffers (CL_MEM_USE_HOST_PTR)
    // =========================================================================
    std::cout << "[Experiment 2: Zero-Copy UMA (CL_MEM_USE_HOST_PTR) (CPU Idle)]\n";
    std::vector<uint16_t> h_A(M * K, 0x3C00);
    std::vector<uint16_t> h_B(K * N, 0x3800);
    std::vector<uint16_t> h_C(M * N, 0);

    cl_mem uma_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bytes_fp16, h_A.data(), nullptr);
    cl_mem uma_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bytes_fp16, h_B.data(), nullptr);
    cl_mem uma_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, bytes_fp16, h_C.data(), nullptr);

    clSetKernelArg(k_fp16, 3, sizeof(cl_mem), &uma_A);
    clSetKernelArg(k_fp16, 4, sizeof(cl_mem), &uma_B);
    clSetKernelArg(k_fp16, 5, sizeof(cl_mem), &uma_C);

    clEnqueueNDRangeKernel(queue, k_fp16, 2, nullptr, g_ws, l_ws, 0, nullptr, nullptr);
    clFinish(queue);

    double uma_ms = 0;
    for (int i = 0; i < 3; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_fp16, 2, nullptr, g_ws, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        uma_ms = (double)(t1 - t0) * 1e-6;
    }
    std::cout << "  Latency: " << uma_ms << " ms | TFLOPS: " << (2.0 * M * N * K / (uma_ms * 1e-3)) * 1e-12 << "\n\n";

    // =========================================================================
    // Test 3: Zero-Copy UMA Buffers WITH CPU Multithreading Active (cTDP Sharing)
    // =========================================================================
    std::cout << "[Experiment 3: Zero-Copy UMA WITH 8 Zen 3 CPU Cores Active (cTDP Contention)]\n";
    std::atomic<bool> cpu_burn{true};
    std::vector<std::thread> workers;
    for (int t = 0; t < 8; t++) {
        workers.emplace_back([&cpu_burn]() {
            __m256 v = _mm256_set1_ps(1.0001f);
            while (cpu_burn.load(std::memory_order_relaxed)) {
                v = _mm256_fmadd_ps(v, v, v);
            }
            volatile float s = ((float*)&v)[0];
            (void)s;
        });
    }

    // Give CPU threads 50 ms to ramp up to 100% load
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    double contended_ms = 0;
    for (int i = 0; i < 3; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_fp16, 2, nullptr, g_ws, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        contended_ms = (double)(t1 - t0) * 1e-6;
    }

    cpu_burn.store(false);
    for (auto& w : workers) w.join();

    std::cout << "  Latency: " << contended_ms << " ms | TFLOPS: " << (2.0 * M * N * K / (contended_ms * 1e-3)) * 1e-12 << "\n";
    std::cout << "  Power Capping Slowdown: " << (contended_ms / uma_ms) << "x slower (" 
              << (1.0 - (uma_ms / contended_ms)) * 100.0 << "% drop in compute throughput!)\n\n";

    return 0;
}
