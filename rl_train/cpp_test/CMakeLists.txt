cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example-app)

set(USE_MKL OFF)

# Set the path to LibTorch
set(CMAKE_PREFIX_PATH "/home/chen2/.local/lib/python3.6/site-packages/torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app model_cpp.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)