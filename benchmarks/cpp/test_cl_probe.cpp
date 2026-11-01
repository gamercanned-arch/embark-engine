#include <CL/cl.h>
#include <iostream>
#include <vector>

int main() {
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::cout << "OpenCL Platforms found: " << num_platforms << std::endl;
    return 0;
}
