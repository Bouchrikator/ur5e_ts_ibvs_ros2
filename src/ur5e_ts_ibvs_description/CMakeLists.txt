cmake_minimum_required(VERSION 3.8)
project(ur5e_ts_ibvs_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes config launch worlds srdf
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/gripper_joint_state_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
