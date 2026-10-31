cmake_minimum_required(VERSION 2.8.8)
project(event_counter)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native -g" )

find_package(catkin REQUIRED COMPONENTS
  roscpp
  dvs_msgs
)

catkin_package(
  INCLUDE_DIRS 
    ${catkin_INCLUDE_DIRS}
    CATKIN_DEPENDS dvs_msgs roscpp
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

list(APPEND SOURCE_FILES
  ../chen_resource_controller/controller/ApplicationMonitor_Appl.cpp
  ../chen_resource_controller/controller/ApplicationMonitor_Common.cpp
)

add_executable(counter ${SOURCE_FILES} src/counter.cc)
target_link_libraries(counter ${catkin_LIBRARIES})
