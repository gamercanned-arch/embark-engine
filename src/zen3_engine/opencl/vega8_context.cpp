// vega8_context.cpp
#include "vega8_context.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace zen3_engine {

const char* Vega8Context::get_error_string(cl_int error) {
    switch (error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: return "CL_UNKNOWN_ERROR";
    }
}

Vega8Context::Vega8Context() {}

Vega8Context::~Vega8Context() {
    cleanup();
}

void Vega8Context::cleanup() {
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    initialized_ = false;
}

bool Vega8Context::init(bool enable_profiling) {
    cleanup();

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "[Vega8Context] Failed to get OpenCL platforms: " << get_error_string(err) << std::endl;
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id selected_platform = nullptr;
    for (cl_uint i = 0; i < num_platforms; ++i) {
        char name[256] = {0};
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        std::string s_name(name);
        if (s_name.find("AMD") != std::string::npos || s_name.find("Accelerated Parallel") != std::string::npos) {
            selected_platform = platforms[i];
            info_.platform_name = s_name;
            break;
        }
    }

    if (!selected_platform) {
        selected_platform = platforms[0];
        char name[256] = {0};
        clGetPlatformInfo(selected_platform, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        info_.platform_name = name;
        std::cout << "[Vega8Context] AMD platform not explicitly matched by name, using: " << info_.platform_name << std::endl;
    } else {
        std::cout << "[Vega8Context] Matched OpenCL Platform: " << info_.platform_name << std::endl;
    }
    platform_ = selected_platform;

    // Discover GPU device
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        std::cerr << "[Vega8Context] No GPU device found in platform: " << get_error_string(err) << std::endl;
        return false;
    }

    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

    cl_device_id selected_device = nullptr;
    for (cl_uint i = 0; i < num_devices; ++i) {
        char dev_name[256] = {0};
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
        std::string s_dev(dev_name);
        if (s_dev.find("gfx90c") != std::string::npos || s_dev.find("Radeon") != std::string::npos) {
            selected_device = devices[i];
            info_.device_name = s_dev;
            break;
        }
    }

    if (!selected_device) {
        selected_device = devices[0];
        char dev_name[256] = {0};
        clGetDeviceInfo(selected_device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
        info_.device_name = dev_name;
    }
    device_ = selected_device;
    std::cout << "[Vega8Context] Matched OpenCL GPU Device: " << info_.device_name << std::endl;

    query_device_info();

    // Create OpenCL context
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_,
        0
    };
    context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_) {
        std::cerr << "[Vega8Context] Failed to create OpenCL context: " << get_error_string(err) << std::endl;
        return false;
    }

    // Create Command Queue with profiling support
    cl_command_queue_properties queue_props = 0;
    if (enable_profiling) {
        queue_props |= CL_QUEUE_PROFILING_ENABLE;
    }

#if defined(CL_VERSION_2_0)
    cl_queue_properties q_props[] = {
        CL_QUEUE_PROPERTIES, (cl_queue_properties)queue_props,
        0
    };
    queue_ = clCreateCommandQueueWithProperties(context_, device_, q_props, &err);
    if (err != CL_SUCCESS) {
        // Fallback to OpenCL 1.2 API
        queue_ = clCreateCommandQueue(context_, device_, queue_props, &err);
    }
#else
    queue_ = clCreateCommandQueue(context_, device_, queue_props, &err);
#endif

    if (err != CL_SUCCESS || !queue_) {
        std::cerr << "[Vega8Context] Failed to create OpenCL command queue: " << get_error_string(err) << std::endl;
        cleanup();
        return false;
    }

    initialized_ = true;
    return true;
}

void Vega8Context::query_device_info() {
    if (!device_) return;

    char driver_ver[256] = {0};
    clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(driver_ver), driver_ver, nullptr);
    info_.driver_version = driver_ver;

    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info_.compute_units), &info_.compute_units, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(info_.max_clock_mhz), &info_.max_clock_mhz, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info_.global_mem_size), &info_.global_mem_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(info_.local_mem_size), &info_.local_mem_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info_.max_work_group_size), &info_.max_work_group_size, nullptr);

    // Check extensions
    size_t ext_size = 0;
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    std::string extensions(ext_size, '\0');
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_size, &extensions[0], nullptr);

    info_.fp16_supported = (extensions.find("cl_khr_fp16") != std::string::npos);

    // Wavefront width on AMD GCN is 64
    info_.wavefront_width = 64;

    std::cout << "[Vega8Context] Compute Units: " << info_.compute_units
              << " | Clock: " << info_.max_clock_mhz << " MHz"
              << " | LDS: " << (info_.local_mem_size / 1024) << " KB"
              << " | Global Mem: " << (info_.global_mem_size / (1024 * 1024)) << " MB"
              << " | FP16 Supported: " << (info_.fp16_supported ? "YES" : "NO")
              << " | Wavefront Width: " << info_.wavefront_width
              << std::endl;
}

cl_program Vega8Context::build_program_from_file(const std::string& cl_filepath, const std::string& build_options) {
    std::ifstream file(cl_filepath);
    if (!file.is_open()) {
        std::cerr << "[Vega8Context] Failed to open OpenCL kernel file: " << cl_filepath << std::endl;
        return nullptr;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return build_program_from_source(ss.str(), build_options);
}

cl_program Vega8Context::build_program_from_source(const std::string& source, const std::string& build_options) {
    if (!initialized_) {
        std::cerr << "[Vega8Context] Cannot build program: Context not initialized!" << std::endl;
        return nullptr;
    }

    const char* src_ptr = source.c_str();
    size_t src_len = source.length();
    cl_int err = CL_SUCCESS;

    cl_program program = clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS || !program) {
        std::cerr << "[Vega8Context] clCreateProgramWithSource failed: " << get_error_string(err) << std::endl;
        return nullptr;
    }

    // Default fast-math and optimization flags for AMD GCN
    std::string opts = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math ";
    opts += build_options;

    err = clBuildProgram(program, 1, &device_, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[Vega8Context] Kernel build failed: " << get_error_string(err) << std::endl;

        // Retrieve build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::cerr << "=== BUILD LOG ===" << std::endl << log << std::endl << "=================" << std::endl;
        }
        clReleaseProgram(program);
        return nullptr;
    }

    return program;
}

cl_kernel Vega8Context::create_kernel(cl_program program, const std::string& kernel_name) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS || !kernel) {
        std::cerr << "[Vega8Context] clCreateKernel('" << kernel_name << "') failed: " << get_error_string(err) << std::endl;
        return nullptr;
    }
    return kernel;
}

cl_mem Vega8Context::wrap_host_buffer(void* host_ptr, size_t size, cl_mem_flags extra_flags) {
    if (!initialized_) {
        throw std::runtime_error("[Vega8Context] Cannot wrap buffer: Context not initialized!");
    }
    if (!host_ptr) {
        throw std::runtime_error("[Vega8Context] Cannot wrap nullptr host pointer!");
    }

    // Zero-Copy UMA mapping: CL_MEM_USE_HOST_PTR
    // On AMD APUs (gfx90c), host memory is mapped directly into GPU virtual address space.
    // This circumvents the 4GB WDDM VRAM allocation limit and avoids host-to-device PCIe copies.
    cl_mem_flags flags = extra_flags | CL_MEM_USE_HOST_PTR;
    cl_int err = CL_SUCCESS;

    cl_mem buffer = clCreateBuffer(context_, flags, size, host_ptr, &err);
    if (err != CL_SUCCESS || !buffer) {
        std::stringstream ss;
        ss << "[Vega8Context] clCreateBuffer(CL_MEM_USE_HOST_PTR) failed for size " << size 
           << " bytes: " << get_error_string(err);
        throw std::runtime_error(ss.str());
    }

    return buffer;
}

cl_mem Vega8Context::allocate_device_buffer(size_t size, cl_mem_flags flags) {
    if (!initialized_) {
        throw std::runtime_error("[Vega8Context] Cannot allocate buffer: Context not initialized!");
    }

    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context_, flags, size, nullptr, &err);
    if (err != CL_SUCCESS || !buffer) {
        std::stringstream ss;
        ss << "[Vega8Context] clCreateBuffer(device) failed for size " << size 
           << " bytes: " << get_error_string(err);
        throw std::runtime_error(ss.str());
    }
    return buffer;
}

void Vega8Context::finish() {
    if (queue_) {
        clFinish(queue_);
    }
}

} // namespace zen3_engine
