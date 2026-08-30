"""Full cable shaping experiment: SOFA truth + estimator, observations,
reduced state, outer TS-PDC, supervisor and (optionally) the inner TS-IBVS.

Launch order follows step 11 of the plan:
  1. Gazebo UR5e + eye-to-hand camera
  2. MoveIt Servo
  3. SOFA truth cable            (owns the physics and the grasp)
  4. Gazebo visual cable         (follows the truth only)
  5. Marker observations         (synthetic or camera)
  6. SOFA estimator              (predicts; never drives the experiment)
  7. Cable state reducer
  8. Cable TS-PDC
  9. Supervisor
 10. Inner eye-to-hand TS-IBVS   (cascade architecture only)

Architectures:
  control_architecture:=direct   outer loop validated on its own
  control_architecture:=cascade  outer loop feeds the inner visual loop, which
                                 stays the ONLY publisher of robot twists.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

CABLE_SHARE = FindPackageShare("cable_identification")
CONTROL_SHARE = FindPackageShare("cable_ts_control")


def _config(package_share, name):
    return PathJoinSubstitution([package_share, "config", name])


def launch_setup(context, *args, **kwargs):
    observation_source = LaunchConfiguration("observation_source").perform(context)
    architecture = LaunchConfiguration("control_architecture").perform(context)
    enable_optimus = LaunchConfiguration("enable_optimus").perform(context)
    enable_control = LaunchConfiguration("enable_control").perform(context)

    if observation_source not in ("synthetic", "camera"):
        raise RuntimeError("observation_source must be 'synthetic' or 'camera'")
    if architecture not in ("direct", "cascade"):
        raise RuntimeError("control_architecture must be 'direct' or 'cascade'")

    use_sim_time = {"use_sim_time": True}
    actions = []

    # 1-2. Robot, camera and Servo. Servo is included explicitly because the
    # optional node inside ur5e_ibvs_sim.launch.py points at a package that
    # does not exist in this workspace.
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ur5e_ts_ibvs_description"),
            "launch", "ur5e_ibvs_sim.launch.py"])),
        launch_arguments={
            "ur_type": LaunchConfiguration("ur_type"),
            "launch_servo": "false",
            "launch_moveit": "false",
            "launch_rviz": LaunchConfiguration("launch_rviz"),
            "gazebo_gui": LaunchConfiguration("gazebo_gui"),
        }.items(),
    ))

    servo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("ibvs_control"), "launch", "servo.launch.py"])),
        condition=IfCondition(LaunchConfiguration("use_servo")),
    )

    # 3. SOFA truth plant: the only owner of the cable physics and the grasp.
    truth = Node(
        package="sofa_ros2_adapter",
        executable="cable_sofa_node",
        name="cable_truth",
        output="screen",
        parameters=[use_sim_time, {
            "role": "truth",
            "cable_config": LaunchConfiguration("truth_config"),
            "base_frame": "base_link",
            "grasp_frame": "cable_grasp_frame",
            "fixture_frame": "cable_fixture_frame",
        }],
    )

    # 4. Visual-only cable in Gazebo.
    visual = Node(
        package="gazebo_cable_visual",
        executable="cable_visual_node",
        output="screen",
        parameters=[use_sim_time, {
            "world": "ibvs_world",
            "world_frame": "world",
            "frames_topic": "/cable/truth/frames",
        }],
    )

    # 5. Observations. Both sources publish /cable/observed_markers, so nothing
    # downstream changes when swapping one for the other.
    if observation_source == "synthetic":
        observations = Node(
            package="cable_perception",
            executable="synthetic_marker_node",
            output="screen",
            parameters=[use_sim_time, {
                "truth_topic": "/cable/truth/markers",
                "noise_std_m": LaunchConfiguration("marker_noise_std_m"),
                "dropout_probability": LaunchConfiguration("marker_dropout"),
                "delay_s": LaunchConfiguration("marker_delay_s"),
                "seed": LaunchConfiguration("seed"),
            }],
        )
    else:
        observations = Node(
            package="cable_perception",
            executable="cable_marker_tracker_node",
            output="screen",
            parameters=[use_sim_time, {
                "image_topic": "/overview/image",
                "camera_info_topic": "/overview/camera_info",
                "camera_frame": "overview_optical_frame",
                "reference_frame": "cable_fixture_frame",
            }],
        )

    # 6. SOFA estimator: same boundary conditions, deliberately wrong
    # parameters, corrected online from the marker observations. It predicts
    # and identifies only; it must never drive the experiment.
    identified = [name.strip() for name in LaunchConfiguration(
        "identified_parameters").perform(context).split(",") if name.strip()]
    estimator = Node(
        package="sofa_ros2_adapter",
        executable="cable_estimator_node",
        name="cable_estimator",
        output="screen",
        condition=IfCondition(enable_optimus),
        parameters=[use_sim_time, {
            "cable_config": LaunchConfiguration("estimator_config"),
            "parameter_bounds_file": LaunchConfiguration("parameter_bounds_file"),
            "identified_parameters": identified,
            "measurement_std_m": LaunchConfiguration("marker_noise_std_m"),
            "base_frame": "base_link",
            "grasp_frame": "cable_grasp_frame",
            "fixture_frame": "cable_fixture_frame",
            "update_rate_hz": 8.0,
        }],
    )

    cable_nodes = [truth, visual, observations, estimator]

    # Validation instrumentation (plan step 12). Independent of the controller
    # so the identification-only runs are recorded the same way.
    cable_nodes.append(Node(
        package="cable_ts_control",
        executable="cable_metrics_recorder_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("record_metrics")),
        parameters=[use_sim_time, {
            "output_csv": LaunchConfiguration("metrics_csv"),
            "truth_config": LaunchConfiguration("truth_config"),
            "identified_parameters": identified,
        }],
    ))

    # 7-9. Outer loop. Needs an identified basis + model, so it is opt-in.
    if enable_control.lower() in ("true", "1"):
        cable_nodes.extend([
            Node(
                package="cable_ts_control",
                executable="cable_state_reducer_node",
                output="screen",
                parameters=[use_sim_time, {
                    "modal_basis_file": LaunchConfiguration("modal_basis_file"),
                    "reference_frame": "cable_fixture_frame",
                }],
            ),
            Node(
                package="cable_ts_control",
                executable="cable_ts_controller_node",
                output="screen",
                parameters=[use_sim_time, {
                    "ts_model_file": LaunchConfiguration("ts_model_file"),
                    "ts_gains_file": LaunchConfiguration("ts_gains_file"),
                    "reference_frame": "cable_fixture_frame",
                    "gripper_frame": "cable_grasp_frame",
                }],
            ),
            Node(
                package="cable_ts_control",
                executable="cable_supervisor_node",
                output="screen",
                parameters=[use_sim_time, {
                    "target_file": LaunchConfiguration("target_config"),
                    "target_modal_coordinates": LaunchConfiguration(
                        "target_modal_coordinates"),
                }],
            ),
        ])

        if architecture == "cascade":
            # The projector is the ONLY bridge to the inner loop: the outer
            # loop never publishes a twist to Servo itself.
            cable_nodes.append(Node(
                package="cable_ts_control",
                executable="cable_reference_projector_node",
                output="screen",
                parameters=[use_sim_time, {
                    "camera_frame": "overview_optical_frame",
                    "marker_size": LaunchConfiguration("marker_size"),
                }],
            ))
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([
                    FindPackageShare("ibvs_control"), "launch", "ibvs.launch.py"])),
                launch_arguments={
                    "controller": "ts_lmi_d",
                    "mode": "sim",
                    "eye_to_hand": "true",
                    "reference_source": "cable",
                    "use_servo": "false",   # started once, above
                    "use_ur": "false",
                    "use_camera": "false",
                }.items(),
            ))

    # Gazebo and TF need a moment before the cable can be clamped to a frame.
    actions.append(TimerAction(period=6.0, actions=[servo] + cable_nodes))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("ur_type", default_value="ur5e"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument("use_servo", default_value="true"),
        DeclareLaunchArgument(
            "observation_source", default_value="synthetic",
            description="synthetic | camera"),
        DeclareLaunchArgument(
            "control_architecture", default_value="direct",
            description="direct | cascade"),
        DeclareLaunchArgument(
            "enable_optimus", default_value="false",
            description="Run the second SOFA instance and identify parameters "
                        "online from the observed markers"),
        DeclareLaunchArgument(
            "identified_parameters", default_value="EI",
            description="Comma-separated parameters to identify; start with EI"),
        DeclareLaunchArgument(
            "enable_control", default_value="false",
            description="Run the outer loop; needs an identified basis + model"),
        DeclareLaunchArgument(
            "truth_config", default_value=_config(CABLE_SHARE, "cable_truth.yaml")),
        DeclareLaunchArgument(
            "estimator_config",
            default_value=_config(CABLE_SHARE, "cable_estimator_initial.yaml")),
        DeclareLaunchArgument(
            "parameter_bounds_file",
            default_value=_config(CONTROL_SHARE, "cable_parameter_bounds.yaml")),
        DeclareLaunchArgument(
            "modal_basis_file",
            default_value=_config(CONTROL_SHARE, "cable_modal_basis.yaml")),
        DeclareLaunchArgument(
            "ts_model_file",
            default_value=_config(CONTROL_SHARE, "cable_ts_model.yaml")),
        DeclareLaunchArgument(
            "ts_gains_file",
            default_value=_config(CONTROL_SHARE, "cable_ts_gains.yaml")),
        DeclareLaunchArgument(
            "target_config",
            default_value=_config(CONTROL_SHARE, "cable_target.yaml")),
        DeclareLaunchArgument(
            "record_metrics", default_value="true",
            description="Write the step 12 validation metrics to CSV"),
        DeclareLaunchArgument(
            "metrics_csv", default_value="/root/ibvs_logs/cable_metrics.csv"),
        DeclareLaunchArgument("target_modal_coordinates", default_value="[0.0, 0.0]"),
        DeclareLaunchArgument("marker_noise_std_m", default_value="0.002"),
        DeclareLaunchArgument("marker_dropout", default_value="0.0"),
        DeclareLaunchArgument("marker_delay_s", default_value="0.0"),
        DeclareLaunchArgument("marker_size", default_value="0.05"),
        DeclareLaunchArgument("seed", default_value="0"),
        OpaqueFunction(function=launch_setup),
    ])
