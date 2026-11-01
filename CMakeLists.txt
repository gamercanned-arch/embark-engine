cmake_minimum_required(VERSION 3.16)
project(EMBARK-Engine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for AMD Zen 3 + AVX2
if (MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast /EHsc /W3)
    add_link_options(/STACK:16777216)
else()
    add_compile_options(-O3 -mavx2 -mf16c -mfma -pthread)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# OpenCL configuration
find_package(OpenCL QUIET)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    set(OPENCL_LIBRARIES ${OpenCL_LIBRARIES})
else()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    if (WIN32)
        set(OPENCL_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/src/zen3_engine/opencl/OpenCL_sys.lib)
    endif()
endif()

# Benchmarks
file(GLOB BENCH_SRCS "benchmarks/cpp/*.cpp")
foreach(BENCH_FILE ${BENCH_SRCS})
    get_filename_component(BENCH_NAME ${BENCH_FILE} NAME_WE)
    # Check if bench links with OpenCL
    file(READ ${BENCH_FILE} FILE_CONTENTS)
    if (FILE_CONTENTS MATCHES "clCreateContext|cl_platform_id")
        add_executable(${BENCH_NAME} ${BENCH_FILE})
        target_link_libraries(${BENCH_NAME} PRIVATE ${OPENCL_LIBRARIES})
    else()
        add_executable(${BENCH_NAME} ${BENCH_FILE})
    endif()
endforeach()

message(STATUS "EMBARK Engine configured for AMD Ryzen 7 7730U APU (Zen 3 + Vega 8 gfx90c)")
