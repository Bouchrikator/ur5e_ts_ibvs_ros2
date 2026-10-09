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
#                ./run.sh cosserat       (restore the Cosserat plugin)
#                ./run.sh optimus        (build/repair patched Cosserat + Optimus)
#   Cable:       ./run.sh cable_plugin_test | cable_forward_test | cable_fom_test
#                ./scripts/run.sh cable_dynamics [--rom] [--drag-steps N] [--tip-offset dx dy]
#                    (every term of M q'' + C q' + f_int = f_g + Jg^T lambda as SOFA
#                     assembles it; --rom adds the POD-Galerkin reduced system M_r, K_r
#                     -> artifacts/cable_dynamics/{npz,png,report.txt,sofa_export*/})
#                ./scripts/run.sh cable_sofa_mor_plugin_test (pinned MOR mapping gate)
#   SOFA MOR:    ./scripts/run.sh cable_sofa_mor_pipeline   (all gates below, in order)
#                cable_sofa_mor_snapshots | cable_sofa_mor_compute_modes |
#                cable_sofa_mor_validate | cable_sofa_modal_dataset |
#                cable_sofa_modal_ts_identify | cable_sofa_modal_lmi |
#                cable_sofa_pdc_test | cable_sofa_modal_observer_test
#                (artifacts in artifacts/cable_mor/, docs/sofa_mor_pipeline.md)
#                ./run.sh optimus_smoke_test    (factory gate, ~2 s)
#                ./run.sh cable_optimus_test    (EI/GJ recovery gates, ~15 s)
#                ./run.sh cable_optimus_pipeline [--duration 60]
#                    (headless truth+markers+estimator, EI must land <5 %)
#                ./run.sh cable_sim      (Gazebo UR5e + SOFA Cosserat cable)
#                ./run.sh cable_identify_sim  (2 SOFA models + EI identification)
#                ./run.sh cable_closed_loop  (truth + estimator + outer loop)
#                ./run.sh cable_qs_test [pose]  (ONE COMMAND: full certified
#                    quasi-static TS-PDC live demo: sim+pose+attach+servo+
#                    approach+latch+relay+Lyapunov monitor)
#                ./run.sh cable_qs_disturb  (tangential pulse on the live loop)
#   Cable ident: ./run.sh cable_dataset -> cable_basis -> cable_identify -> cable_lmi
#                ./run.sh cable_unit_tests
#
# Extra args are forwarded to `ros2 launch ibvs_control ibvs.launch.py`,
# e.g. ./run.sh ts_lmi_d cube_size_x:=0.075 use_servo:=false
# =============================================================================

set -e

# A login session that predates joining the docker group cannot reach
# docker.sock; re-exec under that group once instead of requiring newgrp/re-login.
if [[ -z "$RUN_SH_SG" ]] && ! docker info >/dev/null 2>&1 \
    && getent group docker | grep -qw "$(id -un)"; then
  export RUN_SH_SG=1
  exec sg docker -c "$(printf '%q ' "$0" "$@")"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Allow X11 forwarding
xhost +local:docker 2>/dev/null || true

export RENDER_GID="${RENDER_GID:-$(getent group render | cut -d: -f3)}"

in_container() {
  docker compose run --rm ur5e_ts_ibvs bash -c \
    "source /ros2_ws/install/setup.bash && $*"
}

# Cable commands run in the persistent container (docker compose up -d).
# Cosserat lives in the container overlay, so a `compose down` destroys it;
# ensure_cosserat reinstalls the pinned release (idempotent, ~1 min) instead of
# forcing a full image rebuild.
ensure_cosserat() {
  docker compose exec -T ur5e_ts_ibvs bash /ros2_ws/scripts/install_cosserat.sh
}

# Same idea for the SOFA 25.12 Optimus port plus the source-built Cosserat with
# the EI/GI internal-data patch (third_party/, see docs/optimus_port.md). It is a
# no-op when both markers are present; a container recreated from an image
# without the stage rebuilds them once (~10 min at -j2).
ensure_optimus() {
  docker compose exec -T ur5e_ts_ibvs bash /ros2_ws/scripts/install_optimus.sh
}

# Quiet when healthy, but never let set -e swallow the reason (e.g. docker.sock
# permission denied when the login session predates joining the docker group).
live_up() {
  local out
  out=$(docker compose up -d 2>&1) || { echo "$out" >&2; exit 1; }
}

in_live() {
  live_up
  ensure_cosserat
  ensure_optimus
  if [[ "$CMD" == cable_sofa_* ]]; then
    docker compose exec -T ur5e_ts_ibvs bash /ros2_ws/scripts/install_model_order_reduction.sh
  fi
  # exec bypasses the image entrypoint, so the ROS underlay must be sourced too
  docker compose exec ur5e_ts_ibvs bash -c \
    "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && $*"
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
    # Run from src (mounted rw) so pose edits take effect without a rebuild
    in_live "python3 /ros2_ws/src/ibvs_control/scripts/move_to_view_pose.py ${1:-default} --sim"
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
  cosserat)
    live_up
    ensure_cosserat
    ;;
  optimus)
    live_up
    ensure_cosserat
    ensure_optimus
    ;;
  cable_plugin_test)
    in_live "ros2 run cable_identification cable_plugin_test"
    ;;
  cable_forward_test)
    in_live "ros2 run cable_identification cable_forward_test"
    ;;
  cable_fom_test)
    # numerical verification of the FOM: traction, bending, Jacobian/virtual work, grasp,
    # energy balance, convergence (prints CABLE_FOM_TEST_PASSED)
    in_live "ros2 run cable_identification cable_fom_test $*"
    ;;
  cable_sofa_mor_plugin_test)
    printf -v mor_command '%q ' ros2 run cable_identification cable_sofa_mor_plugin_test "$@"
    in_live "$mor_command"
    ;;
  cable_sofa_mor_snapshots|cable_sofa_mor_compute_modes|cable_sofa_mor_validate|cable_sofa_modal_dataset|cable_sofa_modal_ts_identify|cable_sofa_modal_lmi|cable_sofa_pdc_test|cable_sofa_modal_observer_test|cable_sofa_mor_pipeline)
    printf -v mor_command '%q ' env "CABLE_REPO_COMMIT=$(git rev-parse HEAD)" \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      ros2 run cable_ts_control cable_sofa_mor_pipeline --phase "${CMD#cable_sofa_}" "$@"
    in_live "$mor_command"
    ;;
  optimus_smoke_test)
    in_live "ros2 run cable_identification optimus_smoke_test"
    ;;
  cable_dynamics)
    # Matrices/vectors of the implicit system SOFA solves for the truth cable
    # (--rom needs the pinned ModelOrderReduction mapping, installed by in_live below)
    [[ " $* " == *" --rom "* ]] && CMD=cable_sofa_dynamics
    printf -v dyn_command '%q ' ros2 run cable_identification cable_dynamics_dump "$@"
    in_live "$dyn_command"
    ;;
  cable_optimus_test)
    # Gates D/E/H/F/G of the port protocol: sigma points differ, state restore,
    # log-space bounds, EI and GJ recovery from 0.6x with held-out validation.
    in_live "ros2 run cable_identification optimus_recovery_test $*"
    ;;
  cable_optimus_pipeline)
    # Gate I headless: truth cable + synthetic markers + cable_estimator_node,
    # no GUI, prints OPTIMUS_PIPELINE_TEST_PASSED when EI converges.
    in_live "ros2 run cable_identification optimus_pipeline_test $*"
    ;;
  cable_sim)
    in_live "ros2 launch cable_bringup cable_sim.launch.py $*"
    ;;
  cable_closed_loop)
    in_live "ros2 launch cable_bringup cable_closed_loop_sim.launch.py $*"
    ;;
  cable_qs_test)
    # One-shot certified quasi-static TS-PDC live demo (Gazebo + SOFA GUI):
    # teardown -> sim -> pose -> attach -> servo -> approach/latch -> relay
    in_live "bash /ros2_ws/scripts/cable/qs_live_test.sh $*"
    ;;
  cable_qs_disturb)
    # Tangential disturbance pulse + error/V sampling on a running qs_test
    in_live "python3 /ros2_ws/scripts/cable/disturb_experiment.py $*"
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
      --output /ros2_ws/artifacts/cable_dataset.npz $*"
    ;;
  cable_basis)
    in_live "ros2 run cable_ts_control build_modal_basis \
      --dataset /ros2_ws/artifacts/cable_dataset.npz \
      --output /ros2_ws/artifacts/cable_modal_basis.yaml $*"
    ;;
  cable_identify)
    in_live "ros2 run cable_ts_control identify_ts_vertices \
      --dataset /ros2_ws/artifacts/cable_dataset.npz \
      --basis /ros2_ws/artifacts/cable_modal_basis.yaml \
      --output /ros2_ws/artifacts/cable_ts_model.yaml $*"
    ;;
  cable_lmi)
    in_live "ros2 run cable_ts_control solve_cable_ts_lmi \
      --model /ros2_ws/artifacts/cable_ts_model.yaml \
      --output /ros2_ws/artifacts/cable_ts_gains.yaml $*"
    ;;
  cable_unit_tests)
    in_live "colcon test --packages-select cable_perception cable_ts_control \
      --event-handlers console_direct+ && colcon test-result --verbose"
    ;;
  *)
    echo "Usage: $0 {build|run|up|down|rebuild|test|sim|pose|ts_lmi_d|ts_lmi_c|classic|qmm|eth_<controller>|real|plotjuggler|sofa|cosserat|optimus|cable_plugin_test|cable_forward_test|cable_fom_test|cable_dynamics|optimus_smoke_test|cable_optimus_test|cable_optimus_pipeline|cable_sim|cable_identify_sim|cable_closed_loop|cable_qs_test|cable_qs_disturb|cable_dataset|cable_basis|cable_identify|cable_lmi|cable_unit_tests|cable_sofa_mor_plugin_test|cable_sofa_mor_pipeline|cable_sofa_mor_snapshots|cable_sofa_mor_compute_modes|cable_sofa_mor_validate|cable_sofa_modal_dataset|cable_sofa_modal_ts_identify|cable_sofa_modal_lmi|cable_sofa_pdc_test|cable_sofa_modal_observer_test}"
    exit 1
    ;;
esac
