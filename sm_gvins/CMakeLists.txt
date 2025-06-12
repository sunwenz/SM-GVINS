cmake_minimum_required(VERSION 2.8.3)
project(sm_gvins)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# set(g2o_libs
#     ${PROJECT_SOURCE_DIR}/lib/libg2o_stuff.so
#     ${PROJECT_SOURCE_DIR}/lib/libg2o_core.so
#     ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_dense.so
#     ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_csparse.so
#     ${PROJECT_SOURCE_DIR}/lib/libg2o_csparse_extension.so
#     ${PROJECT_SOURCE_DIR}/lib/libg2o_types_sba.so
#     ${PROJECT_SOURCE_DIR}/lib/libg2o_types_slam3d_addons.so
#     ${PROJECT_SOURCE_DIR}/lib/libg2o_types_slam3d.so
# )
file(GLOB g2o_libs ${PROJECT_SOURCE_DIR}/lib/*.so)

set(glog_lib ${PROJECT_SOURCE_DIR}/thirdparty/glog/install/lib/libglog.a)
# message(STATUS "glog_lib path: ${glog_lib}")

# opencv 4.2.0
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

# yaml-cpp
find_package(yaml-cpp REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/sm_gvins_ros
  ${PROJECT_SOURCE_DIR}/sm_gvins/include/types
  ${PROJECT_SOURCE_DIR}/sm_gvins/include/visual
  ${PROJECT_SOURCE_DIR}/sm_gvins/include/utils
  ${PROJECT_SOURCE_DIR}/sm_gvins/include/
  ${PROJECT_SOURCE_DIR}/thirdparty/sophus
  ${PROJECT_SOURCE_DIR}/thirdparty/eigen
  ${PROJECT_SOURCE_DIR}/thirdparty/g2o
  ${PROJECT_SOURCE_DIR}/thirdparty/g2o/build
  ${PROJECT_SOURCE_DIR}/thirdparty/g2o/EXTERNAL/csparse
  ${PROJECT_SOURCE_DIR}/thirdparty/glog/install/include
)
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

catkin_package(
  LIBRARIES
  CATKIN_DEPENDS roscpp std_msgs cv_bridge
)

file(GLOB VISUAL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/sm_gvins/src/visual/*.cpp
)

add_executable(sm_gvins_kitti 
  test_track.cpp
  sm_gvins/src/estimator.cpp
  sm_gvins/src/imu_preinterg.cpp
  sm_gvins/src/parameters.cpp
  ${VISUAL_SRC}
)

target_link_libraries(sm_gvins_kitti
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${glog_lib}
  ${g2o_libs}
  yaml-cpp
)

# add_executable(sm_gvins_node 
#   test_fronted.cpp
#   ${VISUAL_SRC}
#   sm_gvins/src/estimator.cpp
#   sm_gvins/src/imu_preinterg.cpp
#   sm_gvins/src/parameters.cpp
#   sm_gvins_ros/sm_gvins.cpp
#   sm_gvins_ros/drawer_rviz.cpp
# )

# target_link_libraries(sm_gvins_node
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${glog_lib}
#   ${g2o_libs}
#   yaml-cpp
# )