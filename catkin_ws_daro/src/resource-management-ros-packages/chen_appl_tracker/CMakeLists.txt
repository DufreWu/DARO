cmake_minimum_required(VERSION 3.10)
project(tracker)

# =====================================================
# Dependencies
# =====================================================
find_package(catkin_simple REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

# =====================================================
# Compiler flags
# =====================================================
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -fopenmp"
)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -O3
    --use_fast_math
)

# =====================================================
# Include directories
# =====================================================
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# =====================================================
# Headers
# =====================================================
set(TRACKER_HEADERS
    include/error.h
    include/optimizer.h
    include/patch.h
    include/tracker_cpu.h
    include/tracker_gpu.h
)

# =====================================================
# Sources
# =====================================================
set(TRACKER_CPU_SOURCES
    src/tracker_cpu.cpp
)

set(TRACKER_GPU_SOURCES
    src/tracker_gpu.cpp
)

set(MONITOR_SOURCES
    ../chen_resource_controller/controller/ApplicationMonitor_Appl.cpp
    ../chen_resource_controller/controller/ApplicationMonitor_Common.cpp
)

# =====================================================
# Tracker backend library (CPU + GPU)
# =====================================================
cuda_add_library(tracker_cuda
    src/tracker_gpu_kernels.cu
)

add_library(${PROJECT_NAME}
    ${TRACKER_CPU_SOURCES}
    ${TRACKER_GPU_SOURCES}
    ${MONITOR_SOURCES}
    src/optimizer.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

target_compile_options(${PROJECT_NAME}
    PRIVATE -O3 -fopenmp
)

target_link_libraries(${PROJECT_NAME}
    tracker_cuda
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
)


# =====================================================
# Node executable
# =====================================================
cs_add_executable(tracker_node
    src/tracker_node.cpp
)

target_link_libraries(tracker_node
    ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
)

# =====================================================
# Install & export
# =====================================================
cs_install()
cs_export()
