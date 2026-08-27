cmake_minimum_required(VERSION 3.8)
project(ibvs_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ibvs_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(DEPS rclcpp sensor_msgs geometry_msgs std_msgs cv_bridge tf2_ros ibvs_msgs moveit_msgs)

function(add_controller name src)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${name} ${OpenCV_LIBS} Eigen3::Eigen)
  ament_target_dependencies(${name} ${DEPS})
  install(TARGETS ${name} DESTINATION lib/${PROJECT_NAME})
endfunction()

add_controller(ts_ibvs_continuous_node src/ts_ibvs_continuous_node.cpp)
add_controller(ts_ibvs_discrete_node src/ts_ibvs_discrete_node.cpp)
add_controller(ibvs_classic_node src/ibvs_classic_node.cpp)
add_controller(qmm_mpc_node src/qmm_mpc_node.cpp)
add_controller(eth_reference_node src/eth_reference_node.cpp)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/qmm_mpc_solver_node.py
  scripts/solve_ts_pdc_lmi_discrete.py
  scripts/solve_ts_lmi_reduced.py
  scripts/tp_model_ud_build.py
  scripts/verify_lmis.py
  scripts/move_to_view_pose.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ibvs_math test/test_ibvs_math.cpp)
  target_include_directories(test_ibvs_math PRIVATE include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(test_ibvs_math Eigen3::Eigen ${OpenCV_LIBS})

  ament_add_gtest(test_gains_loader test/test_gains_loader.cpp)
  target_include_directories(test_gains_loader PRIVATE include ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(test_gains_loader Eigen3::Eigen ${OpenCV_LIBS})

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_lmi_synthesis test/test_lmi_synthesis.py
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_package()
