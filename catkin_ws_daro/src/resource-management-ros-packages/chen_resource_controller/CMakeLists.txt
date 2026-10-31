cmake_minimum_required(VERSION 3.10)
project(chen_resource_controller)

SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(
    CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} -lm -lcgroup -pthread -lusb-1.0"
)
SET(
    CMAKE_MODULE_LINKER_FLAGS
    "${CMAKE_MODULE_LINKER_FLAGS} -lm -lcgroup -pthread -lusb-1.0"
)
SET(
    CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -lm -lcgroup -pthread -lusb-1.0"
)
SET(
    CMAKE_STATIC_LINKER_FLAGS
    "${CMAKE_STATIC_LINKER_FLAGS} -lm -lcgroup -pthread -lusb-1.0"
)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -MMD -MP")

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS 
    roscpp 
    std_msgs
)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

catkin_package(
    INCLUDE_DIRS 
    ${catkin_INCLUDE_DIRS}
)

include_directories(
    ${catkin_INCLUDE_DIRS} include
)

file(GLOB SOURCE_FILES "controller/*.cpp")
file(GLOB AGENT_FILES "agents/*.cpp")
message("source files : ${SOURCE_FILES}")
message("source files : ${AGENT_FILES}")

# Set the path to LibTorch
set(CMAKE_PREFIX_PATH "/home/chen2/.local/lib/python3.6/site-packages/torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(controller ${SOURCE_FILES} ${AGENT_FILES} include)
target_link_libraries(controller 
    ${catkin_LIBRARIES} 
    curl
    cgroup 
    Eigen3::Eigen
    "${TORCH_LIBRARIES}"
)
