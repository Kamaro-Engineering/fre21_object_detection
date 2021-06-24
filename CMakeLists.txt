cmake_minimum_required(VERSION 3.0.2)
project(fre_object_detection)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  std_msgs
  sensor_msgs
)

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
install(PROGRAMS
  scripts/fre21_object_detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
