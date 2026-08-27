cmake_minimum_required(VERSION 3.8)
project(ibvs_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(ibvs_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VISP REQUIRED)

add_executable(cube_detector_node src/cube_detector_node.cpp)
target_include_directories(cube_detector_node PRIVATE
  include
  ${VISP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(cube_detector_node ${VISP_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(cube_detector_node
  rclcpp sensor_msgs std_msgs cv_bridge image_transport ibvs_msgs
)

install(TARGETS cube_detector_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_corner_matching test/test_corner_matching.cpp)
  target_include_directories(test_corner_matching PRIVATE include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(test_corner_matching ${OpenCV_LIBS})
endif()

ament_package()
