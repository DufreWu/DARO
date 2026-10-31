cmake_minimum_required(VERSION 3.0.2)
project(edge_image)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp")
add_compile_options(-Wno-unused-variable)

# Use of catkin_simple
find_package(catkin_simple REQUIRED)

# OpenMP is required for CPU code optimization, find it
find_package(OpenMP REQUIRED)
set(OpenMP_LIBS gomp)

# CUDA is required for GPU code, find it
find_package(CUDA REQUIRED)

set(MONITOR
    ../chen_resource_controller/controller/ApplicationMonitor_Appl.cpp
	../chen_resource_controller/controller/ApplicationMonitor_Common.cpp
)

# Find the flowfilter_gpu library
# Be careful: it is installed by default in /usr/local/lib/, but if you changed
# the install path, simply modify it here
find_library(flowfilter_gpu_LIBS NAMES flowfilter_gpu PATHS /usr/local/include)

# Uncomment the three following lines to enable support for .h5 files
# Don't forget to set your own path to correctly find the library
#set(HDF5_DIR "/usr/local/hdf5/cmake")
#find_package(HDF5 REQUIRED COMPONENTS CXX)
#include_directories(${HDF5_INCLUDE_DIR})

# Compile using C+11, -O3 for maximum optimization, enable all error alerts, and add OpenMP flags
add_compile_options(-std=c++11 -O3 -Werror -Wpedantic -Wall -Wextra ${OpenMP_CXX_FLAGS})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 -Xptxas -O3 -Werror all-warnings")

# Initialize catkin_simple
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)
# Edges image computation node, from a live camera
cuda_add_executable(edges_node_live 
    src/edges_node_live.cpp 
    src/edges_cpu.cpp 
    src/edges_gpu.cu 
    src/utils.cpp 
    ${MONITOR}
)

target_link_libraries(edges_node_live 
    ${HDF5_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    ${catkin_LIBRARIES}
    ${OpenMP_LIBS}
)

# Installation
cs_install()

# Export
cs_export()
