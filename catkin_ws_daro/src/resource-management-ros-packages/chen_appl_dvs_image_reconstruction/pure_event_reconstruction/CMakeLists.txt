cmake_minimum_required(VERSION 2.8.3)
project(pure_event_reconstruction)

find_package(catkin_simple REQUIRED)

catkin_simple()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# make libraries
cs_add_library(${PROJECT_NAME} src/pure_event_reconstruction.cpp
							   src/bag_player.cpp
							   src/utils.cpp
							   ../../chen_resource_controller/controller/ApplicationMonitor_Appl.cpp
							   ../../chen_resource_controller/controller/ApplicationMonitor_Common.cpp
							   )

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda-10.2/targets/aarch64-linux/include)

# link required libraries
target_link_libraries(${PROJECT_NAME}
	${catkin_LIBRARIES} 
	${OpenCV_LIBRARIES}
	${CUDA_LIBRARIES}
	${SOURCE_FILES}
)
							   
# list(APPEND SOURCE_FILES
# 	../../chen_resource_controller/controller/ApplicationMonitor_Appl.cpp
# 	../../chen_resource_controller/controller/ApplicationMonitor_Common.cpp
# )



# make the executable
cs_add_executable(reconstruction_node src/reconstruction_node.cpp)

# link required libraries to the executable
target_link_libraries(reconstruction_node ${PROJECT_NAME})

# cs_install()
