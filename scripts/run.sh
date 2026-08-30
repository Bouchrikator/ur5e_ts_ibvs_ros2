#!/bin/bash
# =============================================================================
# UR5 TS-IBVS ROS2 Jazzy — Docker workflow entrypoint
#
#   Container:   ./run.sh build | run | up | down | rebuild | test
#   Simulation:  ./run.sh sim            (Gazebo + UR5 + D435 + cube)
#                ./run.sh pose [name]    (move arm to a viewing pose)
#   Controllers: ./run.sh ts_lmi_d | ts_lmi_c | classic | qmm   [extra args]
#   Eye-to-hand: ./run.sh eth_ts_lmi_d | eth_ts_lmi_c | eth_classic | eth_qmm
#                (fixed overview camera servos the gripper marker above the cube)
#   Real robot:  ./run.sh real ts_lmi_d  [extra args]  (driver+camera+servo)
#   Tools:       ./run.sh plotjuggler
#   SOFA:        ./run.sh sofa [scene]   (host ~/SOFA/scenes → /scenes)
#   Cable:       ./run.sh cable_plugin_test | cable_forward_test
#                ./run.sh cable_sim      (Gazebo UR5e + SOFA Cosserat cable)
#                ./run.sh cable_identify_sim  (2 SOFA models + EI identification)
#                ./run.sh cable_closed_loop  (truth + estimator + outer loop)
#   Cable ident: ./run.sh cable_dataset -> cable_basis -> cable_identify -> cable_lmi
#                ./run.sh cable_unit_tests
#
# Extra args are forwarded to `ros2 launch ibvs_control ibvs.launch.py`,
# e.g. ./run.sh ts_lmi_d cube_size_x:=0.075 use_servo:=false
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Allow X11 forwarding
xhost +local:docker 2>/dev/null || true

in_container() {
  docker compose run --rm ur5e_ts_ibvs bash -c \
    "source /ros2_ws/install/setup.bash && $*"
}

# Cable commands run in the persistent container (docker compose up -d):
# its overlay carries the Cosserat plugin + cable packages until the image
# is rebuilt with ./run.sh build.
in_live() {
  docker compose up -d >/dev/null 2>&1
  docker compose exec ur5e_ts_ibvs bash -c \
    "source /ros2_ws/install/setup.bash && $*"
}

CMD="${1:-build}"
shift || true

case "$CMD" in
  build)
    docker compose build
    ;;
  run)
    docker compose run --rm ur5e_ts_ibvs bash
    ;;
  up)
    docker compose up -d
    docker compose exec ur5e_ts_ibvs bash
    ;;
  down)
    docker compose down
    ;;
  rebuild)
    docker compose run --rm ur5e_ts_ibvs bash -c \
      "source /opt/ros/jazzy/setup.bash && cd /ros2_ws && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"
    ;;
  test)
    docker compose run --rm ur5e_ts_ibvs bash -c \
      "source /opt/ros/jazzy/setup.bash && cd /ros2_ws && colcon test --event-handlers console_direct+ && colcon test-result --verbose"
    ;;
  sim)
    in_container "ros2 launch ur5e_ts_ibvs_description ur5e_ibvs_sim.launch.py $*"
    ;;
  pose)
    in_live "ros2 run ibvs_control move_to_view_pose.py ${1:-default} --sim"
    ;;
  ts_lmi_d|ts_lmi_c|classic|qmm)
    in_container "ros2 launch ibvs_control ibvs.launch.py controller:=$CMD mode:=sim $*"
    ;;
  eth_ts_lmi_d|eth_ts_lmi_c|eth_classic|eth_qmm)
    in_container "ros2 launch ibvs_control ibvs.launch.py controller:=${CMD#eth_} mode:=sim eye_to_hand:=true $*"
    ;;
  real)
    CTRL="${1:-ts_lmi_d}"
    shift || true
    in_container "ros2 launch ibvs_control ibvs.launch.py controller:=$CTRL mode:=real $*"
    ;;
  plotjuggler)
    docker compose exec ur5e_ts_ibvs bash -c \
      "source /ros2_ws/install/setup.bash && ros2 run plotjuggler plotjuggler"
    ;;
  sofa)
    # runSofa wrapper auto-loads SofaPython3 for .py scenes
    docker compose run --rm ur5e_ts_ibvs runSofa "$@"
    ;;
  cable_plugin_test)
    in_live "ros2 run cable_identification cable_plugin_test"
    ;;
  cable_forward_test)
    in_live "ros2 run cable_identification cable_forward_test"
    ;;
  optimus_smoke_test)
    in_live "ros2 run cable_identification optimus_smoke_test"
    ;;
  cable_sim)
    in_live "ros2 launch cable_bringup cable_sim.launch.py $*"
    ;;
  cable_closed_loop)
    in_live "ros2 launch cable_bringup cable_closed_loop_sim.launch.py $*"
    ;;
  cable_identify_sim)
    # Plan milestone: truth + estimator SOFA models, noisy synthetic markers,
    # online EI identification, and deliberately NO controller yet.
    in_live "ros2 launch cable_bringup cable_closed_loop_sim.launch.py \
      enable_optimus:=true enable_control:=false \
      observation_source:=synthetic $*"
    ;;
  cable_dataset)
    # Headless SOFA excitation rollouts -> identification dataset
    in_live "ros2 run cable_ts_control generate_sofa_dataset \
      --config \$(ros2 pkg prefix cable_identification)/share/cable_identification/config/cable_truth.yaml \
      --output /ros2_ws/cable_dataset.npz $*"
    ;;
  cable_basis)
    in_live "ros2 run cable_ts_control build_modal_basis \
      --dataset /ros2_ws/cable_dataset.npz \
      --output /ros2_ws/cable_modal_basis.yaml $*"
    ;;
  cable_identify)
    in_live "ros2 run cable_ts_control identify_ts_vertices \
      --dataset /ros2_ws/cable_dataset.npz \
      --basis /ros2_ws/cable_modal_basis.yaml \
      --output /ros2_ws/cable_ts_model.yaml $*"
    ;;
  cable_lmi)
    in_live "ros2 run cable_ts_control solve_cable_ts_lmi \
      --model /ros2_ws/cable_ts_model.yaml \
      --output /ros2_ws/cable_ts_gains.yaml $*"
    ;;
  cable_unit_tests)
    in_live "colcon test --packages-select cable_perception cable_ts_control \
      --event-handlers console_direct+ && colcon test-result --verbose"
    ;;
  *)
    echo "Usage: $0 {build|run|up|down|rebuild|test|sim|pose|ts_lmi_d|ts_lmi_c|classic|qmm|eth_<controller>|real|plotjuggler|sofa|cable_plugin_test|cable_forward_test|optimus_smoke_test|cable_sim|cable_identify_sim|cable_closed_loop|cable_dataset|cable_basis|cable_identify|cable_lmi|cable_unit_tests}"
    exit 1
    ;;
esac
