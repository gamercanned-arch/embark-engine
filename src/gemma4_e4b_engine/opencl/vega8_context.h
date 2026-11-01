#ifndef VEGA8_GEMMA4_E4B_CONTEXT_H
#define VEGA8_GEMMA4_E4B_CONTEXT_H

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <string>
#include <vector>
#include <iostream>

namespace gemma4_e4b {

struct Vega8DeviceInfo {
    std::string platform_name;
    std::string device_name;
    std::string driver_version;
    cl_uint compute_units = 0;
    cl_uint max_clock_mhz = 0;
    cl_ulong global_mem_size = 0;
    cl_ulong local_mem_size = 0;
    size_t max_work_group_size = 0;
    cl_uint wavefront_width = 0;
    bool fp16_supported = false;
};

class Vega8Context {
public:
    Vega8Context();
    ~Vega8Context();

    Vega8Context(const Vega8Context&) = delete;
    Vega8Context& operator=(const Vega8Context&) = delete;

    bool init(bool enable_profiling = true);

    cl_program build_program_from_file(const std::string& cl_filepath, const std::string& build_options = "");
    cl_program build_program_from_source(const std::string& source, const std::string& build_options = "");

    cl_kernel create_kernel(cl_program program, const std::string& kernel_name);

    // Zero-Copy UMA mapping: Wraps host memory in cl_mem without copying across PCIe
    cl_mem wrap_host_buffer(void* host_ptr, size_t size, cl_mem_flags extra_flags = CL_MEM_READ_ONLY);
    cl_mem allocate_device_buffer(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);

    void finish();

    cl_context get_context() const { return context_; }
    cl_command_queue get_queue() const { return queue_; }
    cl_device_id get_device() const { return device_; }
    const Vega8DeviceInfo& get_device_info() const { return info_; }

    static const char* get_error_string(cl_int error);

private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    Vega8DeviceInfo info_;
    bool initialized_ = false;

    void query_device_info();
    void cleanup();
};

} // namespace spark

#endif // VEGA8_SPARK_CONTEXT_H
