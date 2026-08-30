#!/usr/bin/env python3
"""Unified bringup for the IBVS controllers (sim and real robot).

FETCH pipeline:  camera -> cube_detector -> FeatureTarget -> controller
                 -> TwistStamped -> MoveIt Servo -> UR driver / Gazebo.

Examples
--------
Simulation (start ur5e_ibvs_sim.launch.py first):
  ros2 launch ibvs_control ibvs.launch.py controller:=ts_lmi_d mode:=sim

Real UR5 (driver + RealSense + servo included):
  ros2 launch ibvs_control ibvs.launch.py controller:=ts_lmi_d mode:=real \
      robot_ip:=192.168.1.133
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

CONTROLLERS = {
    # key: (executable, node name, recorder mode)
    "ts_lmi_d": ("ts_ibvs_discrete_node", "ts_ibvs_lmi_discrete"),
    "ts_lmi_c": ("ts_ibvs_continuous_node", "ts_ibvs_lmi"),
    "classic": ("ibvs_classic_node", "ibvs_classic"),
    "qmm": ("qmm_mpc_node", "qmm_ibvs_mpc"),
}


def launch_setup(context, *args, **kwargs):
    controller = LaunchConfiguration("controller").perform(context)
    mode = LaunchConfiguration("mode").perform(context)
    if controller not in CONTROLLERS:
        raise RuntimeError(f"controller must be one of {list(CONTROLLERS)}")
    executable, node_name = CONTROLLERS[controller]

    sim = mode == "sim"
    eth = LaunchConfiguration("eye_to_hand").perform(context).lower() in ("true", "1")
    reference_source = LaunchConfiguration("reference_source").perform(context).lower()
    if reference_source not in ("cube", "cable"):
        raise RuntimeError("reference_source must be 'cube' or 'cable'")
    desired_feature_topic = ("/eth_reference/feature_target"
                             if reference_source == "cube"
                             else "/cable/desired_feature_target")
    use_sim_time = {"use_sim_time": sim}
    share = FindPackageShare("ibvs_control")

    # ---- topics / frames per mode --------------------------------------
    if sim:
        if eth:
            # Fixed scene camera observing cube + gripper marker.
            image_topic = "/overview/image"
            camera_info_topic = "/overview/camera_info"
            depth_topic = "/overview/depth_unused"
            camera_frame = "overview_optical_frame"
        else:
            image_topic = "/d435/color/image"
            camera_info_topic = "/d435/color/camera_info"
            depth_topic = "/d435/depth_image"  # gz depth (32FC1, meters)
            camera_frame = "d435_color_optical_frame"
        use_depth_default = False
    else:
        image_topic = "/camera/color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_frame = "camera_color_optical_frame"
        use_depth_default = not eth  # ETH: fixed camera, area/PnP depth only

    cube_x = ParameterValue(LaunchConfiguration("cube_size_x"), value_type=float)
    cube_y = ParameterValue(LaunchConfiguration("cube_size_y"), value_type=float)
    marker_size = ParameterValue(LaunchConfiguration("marker_size"), value_type=float)

    actions = []

    # ---- perception: cube detector --------------------------------------
    # Eye-in-hand: the cube provides the controlled features.
    # Eye-to-hand: the cube is the GOAL object seen by the fixed camera; the
    # controlled features come from the marker_detector below.
    actions.append(Node(
        package="ibvs_perception",
        executable="cube_detector_node",
        name="cube_detector",
        output="screen",
        parameters=[use_sim_time, {
            "image_topic": image_topic,
            "camera_info_topic": camera_info_topic,
            "depth_topic": depth_topic,
            "use_depth_for_Z": ParameterValue(
                LaunchConfiguration("use_depth_for_z"), value_type=bool) if not eth
                else False,
            "cube_size_x": cube_x,
            "cube_size_y": cube_y,
            # ETH: the cube is small/far in the fixed camera; PnP depth is
            # unreliable there, the area-based estimate is not.
            "prefer_area_depth": eth,
            # QMM historically smoothed corners; others used raw corners.
            "corner_ema_alpha": 0.85 if controller == "qmm" else 0.0,
            "depth_window": 4,
            "depth_min": 0.05,
            "depth_max": 5.0,
        }],
    ))

    if eth:
        # ---- perception: green gripper-marker detector (2nd instance) ----
        actions.append(Node(
            package="ibvs_perception",
            executable="cube_detector_node",
            name="marker_detector",
            output="screen",
            parameters=[use_sim_time, {
                "image_topic": image_topic,
                "camera_info_topic": camera_info_topic,
                "depth_topic": depth_topic,
                "use_depth_for_Z": False,  # plate depth via area estimate
                "prefer_area_depth": True,
                "cube_size_x": marker_size,
                "cube_size_y": marker_size,
                # Green: single hue range (no wrap-around like red).
                "h_low1": 35, "h_high1": 85, "use_hue_range2": False,
                "s_min": 80, "v_min": 80, "min_area": 10.0,
                "corner_ema_alpha": 0.85 if controller == "qmm" else 0.0,
                # TS scheduling range of the ETH setup (marker depth).
                "Z_min": 0.25, "Z_max": 0.85,
            }],
        ))

        # ---- reference generator: cube pose -> desired marker features ----
        # reference_source:=cable hands this job to the outer cable TS loop,
        # which publishes /cable/desired_feature_target instead. Either way a
        # SINGLE reference reaches the controller, and the controller stays
        # the only publisher of /servo_node/delta_twist_cmds.
        if reference_source == "cube":
            actions.append(Node(
                package="ibvs_control",
                executable="eth_reference_node",
                name="eth_reference",
                output="screen",
                parameters=[use_sim_time, {
                    "camera_frame": camera_frame,
                    "base_frame": "base_link",
                    "approach_height": ParameterValue(
                        LaunchConfiguration("approach_height"), value_type=float),
                    "marker_size": marker_size,
                }],
            ))

    # ---- control: selected controller ------------------------------------
    controller_params = [use_sim_time, {
        "cube_size_x": cube_x,
        "cube_size_y": cube_y,
        "command_frame": camera_frame,
        "output_frame": LaunchConfiguration("twist_frame"),
        "servo_topic": "/servo_node/delta_twist_cmds",
    }]
    controller_remaps = []
    if eth:
        # The controller servos the MARKER features toward the moving
        # reference; the twist is mapped v_e = -(cVe)^-1 v_c since the camera
        # is fixed (Chaumette & Hutchinson eq. (6.9)).
        controller_remaps = [("/cube_detector/feature_target",
                              "/marker_detector/feature_target")]
        controller_params.append({
            "desired_feature_topic": desired_feature_topic,
            "eye_to_hand": True,
            "ee_frame": "tool0",
            # ETH depth range (marker as seen by the fixed camera); same
            # values the ROS1 eye-to-hand node used with the same LMI gains.
            "Z0": 0.5, "Z_min": 0.25, "Z_max": 0.85,
        })
    if controller == "ts_lmi_d":
        controller_params.append({"gains_file": LaunchConfiguration("gains_file")})
    actions.append(Node(
        package="ibvs_control",
        executable=executable,
        name=node_name,
        output="screen",
        parameters=controller_params,
        remappings=controller_remaps,
    ))

    # ---- QMM solver service ----------------------------------------------
    if controller == "qmm":
        actions.append(Node(
            package="ibvs_control",
            executable="qmm_mpc_solver_node.py",
            name="qmm_mpc_solver_node",
            output="screen",
            parameters=[use_sim_time, {
                "vertex_file": LaunchConfiguration("vertex_file"),
            }],
        ))

    # ---- MoveIt Servo (twist -> joint trajectory) -------------------------
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "servo.launch.py"])),
        launch_arguments={
            "use_sim_time": str(sim).lower(),
            "command_out_topic":
                "/joint_trajectory_controller/joint_trajectory" if sim else
                "/scaled_joint_trajectory_controller/joint_trajectory",
        }.items(),
        condition=IfCondition(LaunchConfiguration("use_servo")),
    ))

    if not sim:
        # ---- UR driver ----------------------------------------------------
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"])),
            launch_arguments={
                "ur_type": LaunchConfiguration("ur_type"),
                "robot_ip": LaunchConfiguration("robot_ip"),
                "launch_rviz": "false",
                "initial_joint_controller": "scaled_joint_trajectory_controller",
            }.items(),
            condition=IfCondition(LaunchConfiguration("use_ur")),
        ))
        # ---- RealSense D435 -------------------------------------------------
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])),
            launch_arguments={
                "align_depth.enable": "true",
                "enable_depth": "true",
                "rgb_camera.color_profile": "640x480x30",
            }.items(),
            condition=IfCondition(LaunchConfiguration("use_camera")),
        ))
        # ---- tool0 -> camera optical frame mount --------------------------
        # Eye-in-hand only: with a fixed scene camera the user provides the
        # calibrated base_link -> camera TF instead.
        if not eth:
            actions.append(Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="tool0_to_camera_color_optical_frame",
                arguments=[
                    "--x", LaunchConfiguration("mount_x"),
                    "--y", LaunchConfiguration("mount_y"),
                    "--z", LaunchConfiguration("mount_z"),
                    "--roll", LaunchConfiguration("mount_roll"),
                    "--pitch", LaunchConfiguration("mount_pitch"),
                    "--yaw", LaunchConfiguration("mount_yaw"),
                    "--frame-id", "tool0",
                    "--child-frame-id", "camera_color_optical_frame",
                ],
            ))

    return actions


def generate_launch_description():
    share = FindPackageShare("ibvs_control")
    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value="ts_lmi_d",
                              description="ts_lmi_d | ts_lmi_c | classic | qmm"),
        DeclareLaunchArgument("mode", default_value="sim", description="sim | real"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.133"),
        DeclareLaunchArgument("ur_type", default_value="ur5"),
        DeclareLaunchArgument("use_ur", default_value="true",
                              description="Start the UR driver (real mode only)"),
        DeclareLaunchArgument("use_camera", default_value="true",
                              description="Start the RealSense driver (real mode only)"),
        DeclareLaunchArgument("use_servo", default_value="true",
                              description="Start MoveIt Servo"),
        DeclareLaunchArgument("use_depth_for_z", default_value="false",
                              description="Use aligned depth for Z (real camera)"),
        DeclareLaunchArgument("cube_size_x", default_value="0.05"),
        DeclareLaunchArgument("cube_size_y", default_value="0.05"),
        DeclareLaunchArgument("twist_frame", default_value="base_link",
                              description="Frame the commanded twist is re-expressed in "
                                          "(empty = keep camera frame)"),
        DeclareLaunchArgument("eye_to_hand", default_value="false",
                              description="Fixed scene camera: servo the gripper marker "
                                          "to hover approach_height above the cube"),
        DeclareLaunchArgument("reference_source", default_value="cube",
                              description="Source of the desired features (ETH): "
                                          "'cube' = fixed cube reference node, "
                                          "'cable' = outer cable TS loop"),
        DeclareLaunchArgument("approach_height", default_value="0.12",
                              description="Marker goal height above the cube (m, ETH)"),
        DeclareLaunchArgument("marker_size", default_value="0.05",
                              description="Side length of the green gripper marker (m)"),
        DeclareLaunchArgument("mount_x", default_value="0.0"),
        DeclareLaunchArgument("mount_y", default_value="0.0"),
        DeclareLaunchArgument("mount_z", default_value="0.0"),
        DeclareLaunchArgument("mount_roll", default_value="0.0"),
        DeclareLaunchArgument("mount_pitch", default_value="0.0"),
        DeclareLaunchArgument("mount_yaw", default_value="0.0"),
        DeclareLaunchArgument(
            "gains_file",
            default_value=PathJoinSubstitution(
                [share, "config", "ts_pdc_lmi_discrete_gains.yaml"]),
            description="Offline-synthesized TS-PDC gains (discrete controller)"),
        DeclareLaunchArgument(
            "vertex_file",
            default_value=PathJoinSubstitution([share, "config", "tp_vertices.yaml"]),
            description="TP model vertices for the QMM-MPC solver"),
        OpaqueFunction(function=launch_setup),
    ])
