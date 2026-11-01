#include "vega8_context.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace embed_gemma {

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
            break;
        }
    }

    if (!selected_platform) {
        selected_platform = platforms[0];
    }
    platform_ = selected_platform;

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        std::cerr << "[Vega8Context] Failed to get GPU devices: " << get_error_string(err) << std::endl;
        return false;
    }

    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    device_ = devices[0];

    query_device_info();

    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_,
        0
    };

    context_ = clCreateContext(properties, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_) {
        std::cerr << "[Vega8Context] Failed to create OpenCL context: " << get_error_string(err) << std::endl;
        return false;
    }

    cl_command_queue_properties queue_props = 0;
    if (enable_profiling) {
        queue_props |= CL_QUEUE_PROFILING_ENABLE;
    }

    queue_ = clCreateCommandQueue(context_, device_, queue_props, &err);
    if (err != CL_SUCCESS || !queue_) {
        std::cerr << "[Vega8Context] Failed to create command queue: " << get_error_string(err) << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

void Vega8Context::query_device_info() {
    char buf[512] = {0};
    clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(buf), buf, nullptr);
    info_.platform_name = buf;

    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
    info_.device_name = buf;

    clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(buf), buf, nullptr);
    info_.driver_version = buf;

    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &info_.compute_units, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &info_.max_clock_mhz, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &info_.global_mem_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &info_.local_mem_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &info_.max_work_group_size, nullptr);

    cl_uint wave_width = 64;
    clGetDeviceInfo(device_, 0x4037 /* CL_DEVICE_WAVEFRONT_WIDTH_AMD */, sizeof(cl_uint), &wave_width, nullptr);
    info_.wavefront_width = wave_width ? wave_width : 64;

    size_t ext_size = 0;
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    if (ext_size > 0) {
        std::string exts(ext_size, '\0');
        clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_size, &exts[0], nullptr);
        info_.fp16_supported = (exts.find("cl_khr_fp16") != std::string::npos);
    }
}

cl_program Vega8Context::build_program_from_file(const std::string& cl_filepath, const std::string& build_options) {
    std::ifstream file(cl_filepath);
    if (!file.is_open()) {
        std::cerr << "[Vega8Context] Could not open CL file: " << cl_filepath << std::endl;
        return nullptr;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return build_program_from_source(ss.str(), build_options);
}

cl_program Vega8Context::build_program_from_source(const std::string& source, const std::string& build_options) {
    if (!initialized_) return nullptr;

    const char* src_str = source.c_str();
    size_t src_len = source.length();
    cl_int err = CL_SUCCESS;

    cl_program program = clCreateProgramWithSource(context_, 1, &src_str, &src_len, &err);
    if (err != CL_SUCCESS || !program) {
        std::cerr << "[Vega8Context] clCreateProgramWithSource failed: " << get_error_string(err) << std::endl;
        return nullptr;
    }

    std::string full_options = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math " + build_options;

    err = clBuildProgram(program, 1, &device_, full_options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[Vega8Context] clBuildProgram failed: " << get_error_string(err) << std::endl;
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (log_size > 0) {
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::cerr << "[Vega8Context] Build Log:\n" << log << std::endl;
        }
        clReleaseProgram(program);
        return nullptr;
    }

    return program;
}

cl_kernel Vega8Context::create_kernel(cl_program program, const std::string& kernel_name) {
    if (!program) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS || !kernel) {
        std::cerr << "[Vega8Context] Failed to create kernel '" << kernel_name << "': " << get_error_string(err) << std::endl;
        return nullptr;
    }
    return kernel;
}

cl_mem Vega8Context::wrap_host_buffer(void* host_ptr, size_t size, cl_mem_flags extra_flags) {
    if (!initialized_ || !host_ptr || size == 0) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_mem mem = clCreateBuffer(context_, CL_MEM_USE_HOST_PTR | extra_flags, size, host_ptr, &err);
    if (err != CL_SUCCESS || !mem) {
        std::cerr << "[Vega8Context] wrap_host_buffer failed (" << size << " bytes): " << get_error_string(err) << std::endl;
        return nullptr;
    }
    return mem;
}

cl_mem Vega8Context::allocate_device_buffer(size_t size, cl_mem_flags flags) {
    if (!initialized_ || size == 0) return nullptr;
    cl_int err = CL_SUCCESS;
    cl_mem mem = clCreateBuffer(context_, flags, size, nullptr, &err);
    if (err != CL_SUCCESS || !mem) {
        std::cerr << "[Vega8Context] allocate_device_buffer failed: " << get_error_string(err) << std::endl;
        return nullptr;
    }
    return mem;
}

void Vega8Context::finish() {
    if (queue_) {
        clFinish(queue_);
    }
}

} // namespace embed_gemma
