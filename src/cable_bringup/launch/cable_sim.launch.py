"""UR5e + Gazebo Harmonic + SOFA Cosserat cable.

Includes the existing IBVS simulation (ur_type:=ur5e per the cable plan,
Servo/MoveIt off — identification uses prescribed trajectories, not IBVS)
and adds the SOFA cable adapter + the visual-only Gazebo cable.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    cable_config = LaunchConfiguration("cable_config")
    sofa_gui = LaunchConfiguration("sofa_gui")

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur5e_ts_ibvs_description"),
                "launch", "ur5e_ibvs_sim.launch.py"])),
        launch_arguments={
            "ur_type": ur_type,
            "launch_servo": "false",
            "launch_moveit": "false",
            "launch_rviz": launch_rviz,
            "gazebo_gui": gazebo_gui,
        }.items(),
    )

    cable_sofa = Node(
        package="sofa_ros2_adapter",
        executable="cable_sofa_node",
        output="screen",
        condition=UnlessCondition(sofa_gui),
        parameters=[{
            "use_sim_time": True,
            "role": "truth",
            "cable_config": cable_config,
            "base_frame": "base_link",
            "grasp_frame": "cable_grasp_frame",
        }],
    )

    # sofa_gui:=true -> the coupled cable runs inside the runSofa GUI instead
    cable_sofa_gui = ExecuteProcess(
        cmd=["runSofa", "-a",
             PathJoinSubstitution([FindPackageShare("cable_identification"),
                                   "sofa", "cable_scene.py"])],
        output="screen",
        condition=IfCondition(sofa_gui),
        additional_env={"CABLE_CONFIG": ""},
    )

    cable_visual = Node(
        package="gazebo_cable_visual",
        executable="cable_visual_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "world": "ibvs_world",
            "world_frame": "world",
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument("sofa_gui", default_value="false",
                              description="Run the cable in the runSofa GUI"),
        DeclareLaunchArgument(
            "cable_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("cable_identification"),
                "config", "cable_truth.yaml"])),
        sim,
        # give Gazebo + TF a moment before coupling the cable
        TimerAction(period=6.0, actions=[cable_sofa, cable_sofa_gui, cable_visual]),
    ])
