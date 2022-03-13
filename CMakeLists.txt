cmake_minimum_required(VERSION 3.17)
project(segnet)

# Include libraries
find_package(OpenCV 3 REQUIRED)
find_package(Torch REQUIRED PATHS ${PROJECT_SOURCE_DIR}/libtorch/)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)

# Try to compile with c++14
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} supports C++14.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else ()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall")

# Include our header files
include_directories(
        src
        ${OpenCV_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${catkin_LIBRARIES}
)

##################################################
# Make the segmentation library
##################################################
add_library(lib_unet_network
        src/data/CityscapesDataset.cpp
        src/data/Comma10kDataset.cpp
        src/network/blocks/UNetBlocks.h
        src/network/models/DepthModel.h
        src/network/models/UNetModel.h
        src/utils/augmentations.h
)
set_target_properties(lib_unet_network PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(lib_unet_network ${thirdparty_libraries})

##################################################
# Make binary files!
##################################################

add_executable(net_seg_test src/net_seg_test.cpp)
target_link_libraries(net_seg_test lib_unet_network ${thirdparty_libraries})

add_executable(net_seg_train src/net_seg_train.cpp)
target_link_libraries(net_seg_train lib_unet_network ${thirdparty_libraries})

add_executable(test_augments src/test_augments.cpp)
target_link_libraries(test_augments lib_unet_network ${thirdparty_libraries})

add_executable(test_torch src/test_torch.cpp)
target_link_libraries(test_torch lib_unet_network ${thirdparty_libraries})


##################################################
# TODO: build ROS 1 and 2 node!!
##################################################

# Describe catkin Project
#catkin_package(
#    DEPENDS Eigen3 Boost PCL
#    CATKIN_DEPENDS roscpp tf pcl_ros std_msgs geometry_msgs sensor_msgs nav_msgs #visualization_msgs tf_conversions pcl_conversions cv_bridge msckf_init
#    INCLUDE_DIRS src
#)

#add_executable(net_ros src/net_ros.cpp)
#target_link_libraries(net_ros msckf_seg ${thirdparty_libraries})