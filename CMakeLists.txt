cmake_minimum_required(VERSION 3.0.2)
project(coverage_planning)


set(CATKIN_PACKAGE_DEPENDENCIES
  rospy
  sensor_msgs
  std_msgs
)
## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
    ${CATKIN_PACKAGE_DEPENDENCIES}
)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    ${CATKIN_PACKAGE_DEPENDENCIES}
#  DEPENDS system_lib
)


#############
## Install ##
#############

catkin_install_python(PROGRAMS
  cpp/coverage_ros.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)