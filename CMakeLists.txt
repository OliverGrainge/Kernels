cmake_minimum_required(VERSION 3.10)
project(MyCppProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include paths
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set path to LibTorch
set(CMAKE_PREFIX_PATH "/Users/olivergrainge/Documents/github/libtorch/share/cmake")
set(pybind11_DIR "/opt/miniconda3/envs/kernel_env/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

# Find required packages
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# Add the executable (the Python module)
add_library(mymodule SHARED src/quantize.cpp src/bindings.cpp)

# Link against PyTorch and pybind11
target_link_libraries(mymodule "${TORCH_LIBRARIES}" pybind11::module)

