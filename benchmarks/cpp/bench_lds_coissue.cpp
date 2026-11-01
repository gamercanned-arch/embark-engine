#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <iomanip>
#include <vector>

const char* LDS_DUAL_ISSUE_SRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LDS_PAD 2

// Test 1: Pure Register FMA Loop (RPM) - measures pure ALU ceiling with no memory
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void test_pure_alu_rpm(const int iters, __global half* out) {
    const int tid = get_local_id(1) * 8 + get_local_id(0);

    half2 c[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = (half2)((half)tid, (half)i);
    }
    half2 a = (half2)(1.001h, 0.999h);
    half2 b = (half2)(1.002h, 0.998h);

    for (int k = 0; k < iters; k++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                c[i][j] = fma(a, b, c[i][j]);
            }
        }
    }

    half sum = 0.0h;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) sum += c[i][j].x + c[i][j].y;
    }
    out[tid] = sum;
}

// Test 2: Dual-Issued LDS Read + FMA Loop
// Each step k reads LDS into a temporary and computes 32 FMAs
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void test_lds_fma_interleaved(const int iters, __global half* out) {
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tid = ty * 8 + tx;

    __local half2 lds_b[32][32 + LDS_PAD];
    // Init LDS
    for (int i = 0; i < 16; i++) {
        lds_b[ty * 2 + (i >> 3)][tx * 4 + (i & 3)] = (half2)(1.001h, 0.999h);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    half2 c[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = (half2)((half)tid, (half)i);
    }
    half2 a = (half2)(1.001h, 0.999h);

    for (int k = 0; k < iters; k++) {
        int lds_idx = (k & 31);
        half2 b0 = lds_b[lds_idx][tx * 4 + 0];
        half2 b1 = lds_b[lds_idx][tx * 4 + 1];
        half2 b2 = lds_b[lds_idx][tx * 4 + 2];
        half2 b3 = lds_b[lds_idx][tx * 4 + 3];

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            c[i][0] = fma(a, b0, c[i][0]);
            c[i][1] = fma(a, b1, c[i][1]);
            c[i][2] = fma(a, b2, c[i][2]);
            c[i][3] = fma(a, b3, c[i][3]);
        }
    }

    half sum = 0.0h;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) sum += c[i][j].x + c[i][j].y;
    }
    out[tid] = sum;
}
)CLC";

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  GCN 5.0 Hardware Limits: Pure ALU vs LDS Co-Issue Ceilings    \n";
    std::cout << "=================================================================\n";

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, props, nullptr);

    size_t len = strlen(LDS_DUAL_ISSUE_SRC);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &LDS_DUAL_ISSUE_SRC, &len, nullptr);
    cl_int b_err = clBuildProgram(prog, 1, &device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
    if (b_err != CL_SUCCESS) {
        size_t lsz;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        std::cerr << "Build Log:\n" << log.data() << std::endl;
        return 1;
    }

    cl_kernel k_alu = clCreateKernel(prog, "test_pure_alu_rpm", nullptr);
    cl_kernel k_lds = clCreateKernel(prog, "test_lds_fma_interleaved", nullptr);

    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 64 * sizeof(uint16_t), nullptr, nullptr);

    int iters = 200000;
    clSetKernelArg(k_alu, 0, sizeof(int), &iters);
    clSetKernelArg(k_alu, 1, sizeof(cl_mem), &d_out);

    clSetKernelArg(k_lds, 0, sizeof(int), &iters);
    clSetKernelArg(k_lds, 1, sizeof(cl_mem), &d_out);

    // Launch across 8 CUs (e.g. 64 workgroups of 64 threads = 4096 threads)
    size_t g_ws[2] = { 64, 64 }; // 4096 threads
    size_t l_ws[2] = { 8, 8 };   // 64 threads per workgroup

    // Warmup
    clEnqueueNDRangeKernel(queue, k_alu, 2, nullptr, g_ws, l_ws, 0, nullptr, nullptr);
    clFinish(queue);

    // FLOPs per thread = iters * 32 FMAs * 4 FLOPs = iters * 128 FLOPs.
    // Total FLOPs = 4096 * iters * 128.
    double total_flops = 4096.0 * (double)iters * 128.0;

    // Test 1: Pure ALU
    double best_alu_ms = 1e9;
    for (int i = 0; i < 5; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_alu, 2, nullptr, g_ws, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        double ms = (double)(t1 - t0) * 1e-6;
        if (ms < best_alu_ms) best_alu_ms = ms;
    }
    double alu_tflops = (total_flops / (best_alu_ms * 1e-3)) * 1e-12;
    std::cout << "  Pure ALU Register RPM Ceiling  : " << best_alu_ms << " ms | " << alu_tflops << " TFLOPS\n";

    // Test 2: Interleaved LDS Reads + ALU
    double best_lds_ms = 1e9;
    for (int i = 0; i < 5; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, k_lds, 2, nullptr, g_ws, l_ws, 0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        clReleaseEvent(ev);
        double ms = (double)(t1 - t0) * 1e-6;
        if (ms < best_lds_ms) best_lds_ms = ms;
    }
    double lds_tflops = (total_flops / (best_lds_ms * 1e-3)) * 1e-12;
    std::cout << "  Co-Issued LDS Read + ALU RPM   : " << best_lds_ms << " ms | " << lds_tflops << " TFLOPS\n";

    return 0;
}
