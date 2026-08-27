"""
Launch UR5 + Robotiq 85 + RealSense D435 in Gazebo Harmonic.

Replicates the official ur_simulation_gz/ur_sim_control.launch.py pattern
with one fix: ParameterValue wrapper for robot_description (needed because
our extended URDF triggers the YAML parser — upstream does not need it).

Additions over the official launch:
  - D435 camera topic bridge (Gazebo transport → ROS 2)
  - Optional MoveIt Servo node for TS-IBVS

Usage:
  ros2 launch ur5e_ts_ibvs_description ur5e_ibvs_sim.launch.py
  ros2 launch ur5e_ts_ibvs_description ur5e_ibvs_sim.launch.py launch_servo:=true
"""

import os
import yaml
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    # ---- Arguments ----
    ur_type = LaunchConfiguration("ur_type")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    controllers_file = LaunchConfiguration("controllers_file")
    tf_prefix = LaunchConfiguration("tf_prefix")
    activate_joint_controller = LaunchConfiguration("activate_joint_controller")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    description_file = LaunchConfiguration("description_file")
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_servo = LaunchConfiguration("launch_servo")
    launch_moveit = LaunchConfiguration("launch_moveit")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    world_file = LaunchConfiguration("world_file")

    # ---- Robot description (xacro → URDF string) ----
    # Identical to official ur_sim_control.launch.py
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ", description_file,
            " ", "safety_limits:=", safety_limits,
            " ", "safety_pos_margin:=", safety_pos_margin,
            " ", "safety_k_position:=", safety_k_position,
            " ", "name:=", "ur",
            " ", "ur_type:=", ur_type,
            " ", "tf_prefix:=", tf_prefix,
            " ", "simulation_controllers:=", controllers_file,
        ]
    )
    # ParameterValue wrapper — needed because our extended URDF (D435 + gripper
    # meshes) contains characters that confuse the YAML parameter parser.
    # The official launch omits this because their simpler URDF doesn't trigger it.
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # ---- Nodes (same as official ur_sim_control.launch.py) ----
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    # ---- Plain RViz (no MoveIt) ----
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", PathJoinSubstitution(
            [FindPackageShare("ur_description"), "rviz", "view_robot.rviz"])],
        condition=IfCondition(launch_rviz),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                    "--controller-manager", "/controller_manager",
                    "--controller-manager-timeout", "120"],
    )

    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[rviz_node],
        ),
        condition=IfCondition(launch_rviz),
    )

    # ---- MoveIt move_group + RViz (when launch_moveit:=true) ----
    # Use our custom SRDF that extends official UR SRDF with
    # disable_collisions for gripper, camera, and pedestal links.
    our_srdf = Path(
        get_package_share_directory("ur5e_ts_ibvs_description")
    ) / "srdf" / "ur5e_ibvs_setup.srdf.xacro"
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description_semantic(our_srdf, {"name": "ur"})
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True,
             "publish_robot_description_semantic": True},
        ],
        condition=IfCondition(launch_moveit),
    )

    moveit_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", PathJoinSubstitution(
            [FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"])],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
        condition=IfCondition(launch_moveit),
    )

    wait_robot_description = Node(
        package="ur_robot_driver",
        executable="wait_for_robot_description",
        output="screen",
        condition=IfCondition(launch_moveit),
    )

    delay_moveit_after_robot_description = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=wait_robot_description,
            on_exit=[move_group_node, moveit_rviz_node],
        ),
        condition=IfCondition(launch_moveit),
    )

    initial_joint_controller_spawner_started = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[initial_joint_controller, "-c", "/controller_manager",
                   "--controller-manager-timeout", "120"],
        condition=IfCondition(activate_joint_controller),
    )

    initial_joint_controller_spawner_stopped = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[initial_joint_controller, "-c", "/controller_manager",
                   "--stopped",
                   "--controller-manager-timeout", "120"],
        condition=UnlessCondition(activate_joint_controller),
    )

    # ---- Gazebo (identical to official) ----
    # Use -topic instead of -string because our extended URDF (36KB with gripper,
    # camera, pedestal, table) is too large for reliable command-line arg passing.
    # The official UR URDF is small enough for -string, ours is not.
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "ur",
            "-allow_renaming", "true",
        ],
    )

    # Server and GUI as separate processes: the combined `gz sim -r` fork
    # mode hangs inside Docker (server child never starts, `create` loops on
    # "Requesting list of world names"). Split mode works reliably.
    gz_server_launch_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={
            "gz_args": [" -s -r -v 4 ", world_file],
        }.items(),
    )

    gz_gui_launch_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": " -g "}.items(),
        condition=IfCondition(gazebo_gui),
    )

    gz_clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    # ---- Our additions (D435 + overview camera bridge + MoveIt Servo) ----
    gz_camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/d435/color/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/d435/color/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/d435/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/d435/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/d435/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            # Fixed scene camera (eye-to-hand)
            "/overview/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/overview/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
    )

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("ur5e_ts_ibvs"), "config", "servo_config.yaml"]
            ),
            {"use_sim_time": True},
        ],
        condition=IfCondition(launch_servo),
    )

    # ---- Gripper joint state publisher ----
    # Publishes default (0) positions for gripper joints not managed by
    # gz_ros2_control, so MoveIt's planning_scene_monitor has full state.
    gripper_joint_state_publisher = Node(
        package="ur5e_ts_ibvs_description",
        executable="gripper_joint_state_publisher.py",
        name="gripper_joint_state_publisher",
        output="log",
        parameters=[{"use_sim_time": True}],
    )

    # ---- Launch order (matches official) ----
    nodes_to_start = [
        robot_state_publisher_node,
        gripper_joint_state_publisher,
        joint_state_broadcaster_spawner,
        delay_rviz_after_joint_state_broadcaster_spawner,
        initial_joint_controller_spawner_stopped,
        initial_joint_controller_spawner_started,
        gz_spawn_entity,
        gz_server_launch_description,
        gz_gui_launch_description,
        gz_clock_bridge,
        gz_camera_bridge,
        servo_node,
        # MoveIt
        wait_robot_description,
        delay_moveit_after_robot_description,
    ]

    return nodes_to_start


def generate_launch_description():
    desc_pkg = FindPackageShare("ur5e_ts_ibvs_description")

    declared_arguments = [
        DeclareLaunchArgument("ur_type", default_value="ur5",
            description="UR robot type",
            choices=["ur3", "ur5", "ur10", "ur3e", "ur5e", "ur7e",
                     "ur10e", "ur12e", "ur16e", "ur20", "ur30"]),
        DeclareLaunchArgument("safety_limits", default_value="true"),
        DeclareLaunchArgument("safety_pos_margin", default_value="0.15"),
        DeclareLaunchArgument("safety_k_position", default_value="20"),
        DeclareLaunchArgument("tf_prefix", default_value='""'),
        DeclareLaunchArgument("controllers_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "config",
                 "ur_controllers.yaml"]),
            description="Controllers YAML (default: official ur_simulation_gz)"),
        DeclareLaunchArgument("description_file",
            default_value=PathJoinSubstitution(
                [desc_pkg, "urdf", "ur5e_ibvs_setup.urdf.xacro"]),
            description="URDF xacro file"),
        DeclareLaunchArgument("world_file",
            default_value=PathJoinSubstitution(
                [desc_pkg, "worlds", "ibvs_world.sdf"]),
            description="Gazebo world SDF (default: ibvs_world.sdf with cube)"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("launch_servo", default_value="false",
            description="Launch MoveIt Servo node"),
        DeclareLaunchArgument("launch_moveit", default_value="false",
            description="Launch MoveIt move_group + RViz with motion planning"),
        DeclareLaunchArgument("activate_joint_controller", default_value="true"),
        DeclareLaunchArgument("initial_joint_controller",
            default_value="joint_trajectory_controller",
            description="Robot controller to start (gz sim has no scaled_ "
                        "controller; that one is real-hardware only)."),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
