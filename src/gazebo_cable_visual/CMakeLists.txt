cmake_minimum_required(VERSION 3.16)
project(gazebo_cable_visual)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport REQUIRED)
find_package(gz_msgs_vendor REQUIRED)
find_package(gz-msgs REQUIRED)

add_executable(cable_visual_node src/cable_visual_node.cpp)
ament_target_dependencies(cable_visual_node rclcpp geometry_msgs tf2_ros)
target_link_libraries(cable_visual_node gz-transport::core gz-msgs::gz-msgs)

install(TARGETS cable_visual_node DESTINATION lib/${PROJECT_NAME})

ament_package()
