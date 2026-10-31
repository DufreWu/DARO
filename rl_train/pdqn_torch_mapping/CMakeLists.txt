cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(pdqn_app)

set(USE_MKL OFF)

# Set the path to LibTorch
set(CMAKE_PREFIX_PATH "/home/chen2/.local/lib/python3.6/site-packages/torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pdqn_app pdqn_cpp.cpp)
target_link_libraries(pdqn_app "${TORCH_LIBRARIES}")
set_property(TARGET pdqn_app PROPERTY CXX_STANDARD 17)