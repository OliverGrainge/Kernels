cmake_minimum_required(VERSION 3.12)
project(VectorAddExample LANGUAGES CXX CUDA)

# Add include directory (adjust the path if needed)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files for CPU and GPU implementations
set(CPU_SOURCES src/cpu/vectorAdd_cpu.cpp)
set(GPU_SOURCES src/gpu/vectorAdd_gpu.cu)

# Find all example source files in the src/examples directory
file(GLOB EXAMPLE_SOURCES src/examples/*.cpp)

# Iterate over each example source file and create an executable
foreach(EXAMPLE_FILE ${EXAMPLE_SOURCES})
    # Extract the filename without extension to use as the target name
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)

    # Create an executable for each example
    add_executable(${EXAMPLE_NAME} ${CPU_SOURCES} ${GPU_SOURCES} ${EXAMPLE_FILE})

    # Specify CUDA architecture (optional)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES CUDA_ARCHITECTURES "60;70;75;80")

    # Enable C++11 (or higher)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES CXX_STANDARD 11)
endforeach()
