#!/usr/bin/env python3
"""MoveIt Servo bringup for the UR5 (twist -> joint trajectory).

Builds the plain UR description (ur_description + ur_moveit_config SRDF) so the
same servo instance works for the Gazebo sim and the real driver — both publish
/joint_states for the same six joints.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def load_yaml(package: str, relative_path: str):
    path = os.path.join(get_package_share_directory(package), relative_path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type")

    robot_description = {
        "robot_description": ParameterValue(
            Command([
                PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
                PathJoinSubstitution(
                    [FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"]),
                " ", "ur_type:=", ur_type, " ", "name:=ur",
            ]),
            value_type=str),
    }
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            Command([
                PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
                PathJoinSubstitution(
                    [FindPackageShare("ur_moveit_config"), "srdf", "ur.srdf.xacro"]),
                " ", "name:=ur",
            ]),
            value_type=str),
    }
    kinematics = {
        "robot_description_kinematics":
            load_yaml("ur_moveit_config", "config/kinematics.yaml"),
    }
    joint_limits = {
        "robot_description_planning":
            load_yaml("ur_moveit_config", "config/joint_limits.yaml"),
    }
    servo_params = {
        "moveit_servo": load_yaml("ibvs_control", "config/servo_config.yaml"),
    }

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
            {"moveit_servo.command_out_topic": LaunchConfiguration("command_out_topic")},
            {"use_sim_time": ParameterValue(
                LaunchConfiguration("use_sim_time"), value_type=bool)},
            robot_description,
            robot_description_semantic,
            kinematics,
            joint_limits,
        ],
    )
    return [servo_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur5"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "command_out_topic",
            default_value="/joint_trajectory_controller/joint_trajectory"),
        OpaqueFunction(function=launch_setup),
    ])
