cmake_minimum_required(VERSION 3.0.2)
project(realtime_detection_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  DetectedObject.msg
  DetectedObjects.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

## The catkin_package macro generates cmake config files for your package
catkin_package(
  CATKIN_DEPENDS
  std_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
)

## Build
## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
) 