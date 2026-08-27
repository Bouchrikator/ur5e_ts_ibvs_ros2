#!/bin/bash
# =============================================================================
# UR5 TS-IBVS ROS2 Jazzy — Docker workflow entrypoint
#
#   Container:   ./run.sh build | run | up | down | rebuild | test
#   Simulation:  ./run.sh sim            (Gazebo + UR5 + D435 + cube)
#                ./run.sh pose [name]    (move arm to a viewing pose)
#   Controllers: ./run.sh ts_lmi_d | ts_lmi_c | classic | qmm   [extra args]
#   Eye-to-hand: ./run.sh pose eth   (marker faces the scene camera)
#                ./run.sh eth_ts_lmi_d | eth_ts_lmi_c | eth_classic | eth_qmm
#                (fixed overview camera servos the gripper marker above the cube)
#   Real robot:  ./run.sh real ts_lmi_d  [extra args]  (driver+camera+servo)
#   Tools:       ./run.sh plotjuggler
#   SOFA:        ./run.sh sofa [scene]   (host ~/SOFA/scenes → /scenes)
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
    in_container "ros2 run ibvs_control move_to_view_pose.py ${1:-default} --sim"
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
  *)
    echo "Usage: $0 {build|run|up|down|rebuild|test|sim|pose|ts_lmi_d|ts_lmi_c|classic|qmm|eth_<controller>|real|plotjuggler|sofa}"
    exit 1
    ;;
esac
