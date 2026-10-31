cmake_minimum_required(VERSION 3.0.2)
project(fa_harris)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp" )
add_compile_options(-Wno-unused-variable)

# Add -O0 to remove optimizations when using gcc
# IF(CMAKE_COMPILER_IS_GNUCC)
#     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
#     set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
# ENDIF(CMAKE_COMPILER_IS_GNUCC)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  dvs_msgs
  cv_bridge
)

catkin_package(
  INCLUDE_DIRS 
    ${catkin_INCLUDE_DIRS}
)

include_directories(
  include/fa_harris/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)
message("Opencv lib : ${OpenCV_INCLUDE_DIRS}")
set(LIBS ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${YAML_CPP_LIBRARIES})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 -Xptxas -O3 -Werror all-warnings")
add_definitions(-D_FORCE_INLINES)  # Fix CUDA inline function issues

list(APPEND HEADER_FILES
  include/fa_harris/include/fa_harris_detector_cpu.h
  include/fa_harris/include/fa_harris_detector_gpu.h
)

list(APPEND SOURCE_FILES
  include/fa_harris/src/fa_harris_detector_cpu.cpp
  include/fa_harris/src/fa_harris_detector_gpu.cu
  ../chen_resource_controller/controller/ApplicationMonitor_Appl.cpp
  ../chen_resource_controller/controller/ApplicationMonitor_Common.cpp
)

cuda_add_executable(fa_harris ${SOURCE_FILES} src/fa_harris.cpp)
target_link_libraries(fa_harris ${LIBS})
