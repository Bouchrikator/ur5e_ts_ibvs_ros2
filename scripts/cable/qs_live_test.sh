#!/bin/bash
# =============================================================================
# One-command live demo of the certified quasi-static TS-PDC (audit §12).
# Runs INSIDE the container:  bash /ros2_ws/scripts/cable/qs_live_test.sh [pose]
# From the host:              ./scripts/run.sh cable_qs_test [pose]
#
# Sequence (mirrors the verified 2026-08-30 run):
#   1. tear down any previous stack
#   2. Gazebo + SOFA closed-loop sim (GUI) with the certified narrow artifacts
#   3. disable the supervisor (terminal-FAULT design) and the inner IBVS node
#   4. pose the arm at the cable grasp point, attach the SOFA cable
#   5. MoveIt Servo in TWIST mode (the launch include never spawns it)
#   6. approach3: drive gripper to the basis boundary reference, latch q_eq
#   7. twist_relay: certified outer twist -> Servo (TF fixture -> base_link)
#   8. foreground Lyapunov monitor; Ctrl-C detaches, stack keeps running
# =============================================================================
set -e

POSE="${1:-cable_high}"
ART=/ros2_ws/artifacts
HELPERS=/ros2_ws/scripts/cable

source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
export SOFA_ROOT=/opt/sofa
export PYTHONPATH=/opt/sofa/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH

log() { echo -e "\e[1;36m[qs_live_test]\e[0m $*"; }
die() { echo -e "\e[1;31m[qs_live_test] FAIL: $*\e[0m" >&2; exit 1; }

wait_for() {  # wait_for <description> <deadline_s> <check-fn...>
  local desc=$1 dl=$((SECONDS + $2)); shift 2
  until "$@"; do
    (( SECONDS > dl )) && { tail -40 /tmp/cable_sim.log 2>/dev/null; die "timeout waiting for $desc"; }
    sleep 2
  done
  log "$desc: ready"
}
topic_alive() { timeout 6 ros2 topic echo "$1" --once >/dev/null 2>&1; }
jtc_ready() {
  [[ "$(ros2 topic info /joint_trajectory_controller/joint_trajectory 2>/dev/null \
        | awk '/Subscription count/{print $3}')" == [1-9] ]]
}
servo_ready() { ros2 service list 2>/dev/null | grep -q "^/servo_node/switch_command_type$"; }
grasped() {
  [[ "$(timeout 6 ros2 topic echo /cable/grasp_state --once --field state 2>/dev/null | head -1)" == "2" ]]
}

# --- 0. preconditions --------------------------------------------------------
[[ -f $ART/cable_qs_gains_narrow.yaml ]] || die "missing certified gains $ART/cable_qs_gains_narrow.yaml"
avail_gb=$(free -g | awk '/^Mem:/{print $7}')
if (( avail_gb < 3 )); then
  log "WARNING: only ${avail_gb} GB RAM available - Gazebo can OOM silently; close browsers etc."
fi

# --- 1. teardown any previous stack (bracket patterns: no self-match) --------
log "tearing down any previous stack"
for pat in "[c]able_sofa_node" "[c]able_estimator_node" "[c]able_state_reducer" \
           "[c]able_ts_controller" "[c]able_supervisor" "[c]able_reference_projector" \
           "[c]able_visual_node" "[c]able_dlo_detector" "[c]able_marker_tracker" \
           "[c]able_metrics_recorder" "[s]ynthetic_marker_node" "[t]s_ibvs" \
           "[s]ervo_node" "[t]wist_relay" "[a]pproach3" "[d]isturb_experiment" \
           "[m]ove_to_view_pose" "[r]os2 launch" "[s]pawner" "[r]viz2" \
           "[c]ube_detector" "[e]th_reference" "[g]ripper_joint_state" \
           "[s]tatic_transform_publisher" "[q]mm_mpc" \
           "[r]obot_state_publisher" "[p]arameter_bridge" "[i]mage_bridge" "[g]z sim"; do
  pkill -f "$pat" 2>/dev/null || true
done
sleep 3
pkill -9 -f "[g]z sim" 2>/dev/null || true
sleep 1

# --- 2. Gazebo + SOFA closed loop with the certified artifacts ---------------
log "launching Gazebo + SOFA closed loop (GUI) with certified narrow artifacts"
# use_servo:=false — servo streams hold trajectories that preempt the posing
# step; the demo starts servo itself right after attach (step 5).
setsid ros2 launch cable_bringup cable_closed_loop_sim.launch.py \
  observation_source:=camera enable_optimus:=false enable_control:=true \
  control_architecture:=cascade gazebo_gui:=true use_servo:=false \
  modal_basis_file:=$ART/cable_modal_basis_narrow.yaml \
  ts_model_file:=$ART/cable_qs_model_narrow.yaml \
  ts_gains_file:=$ART/cable_qs_gains_narrow.yaml \
  </dev/null >/tmp/cable_sim.log 2>&1 &

wait_for "Gazebo joint states" 180 topic_alive /joint_states
wait_for "SOFA cable (grasp_state)" 120 topic_alive /cable/grasp_state
wait_for "joint_trajectory_controller subscriber" 120 jtc_ready

# --- 3. verified workaround: the inner IBVS node fights the relay on the
#        servo topic. The supervisor now adopts external targets and
#        recovers from FAULT, so it stays alive. ------------------------------
log "disabling inner IBVS node (verified workaround)"
pkill -f "[t]s_ibvs_discrete_node" 2>/dev/null || true

# --- 4. pose the arm and attach the cable (servo NOT running yet) ------------
log "posing arm to '$POSE'"
python3 /ros2_ws/src/ibvs_control/scripts/move_to_view_pose.py "$POSE" --sim
sleep 8

log "requesting cable attach"
ros2 service call /cable/attach std_srvs/srv/Trigger >/dev/null
wait_for "grasp latched (state=2; if stuck, retry with pose cable_low)" 30 grasped

# --- 5. MoveIt Servo in twist mode --------------------------------------------
# The launch include is fixed (ur_type/use_sim_time now passed), so servo may
# already be up; only start one when it is not.
if servo_ready; then
  log "servo_node already up (launch include)"
else
  log "starting MoveIt Servo"
  setsid ros2 launch ibvs_control servo.launch.py ur_type:=ur5e use_sim_time:=true \
    </dev/null >/tmp/servo.log 2>&1 &
  wait_for "servo_node" 90 servo_ready
fi
sleep 2
ros2 service call /servo_node/switch_command_type \
  moveit_msgs/srv/ServoCommandType "{command_type: 1}" >/dev/null
log "servo in TWIST mode"

# --- 6. approach the premise box and latch the settled equilibrium -----------
log "approach: driving gripper to basis boundary reference, then latching q_eq"
timeout 420 python3 $HELPERS/approach3.py || \
  die "approach did not settle/latch (is the cable visible to the overview camera?)"

# --- 7. certified outer loop -> Servo ------------------------------------------
log "starting certified twist relay"
setsid python3 $HELPERS/twist_relay.py </dev/null >/tmp/relay.log 2>&1 &
sleep 2

# --- 8. monitor ---------------------------------------------------------------
log "closed loop ACTIVE - certified QS TS-PDC with manifold bias anchor"
log "logs: /tmp/cable_sim.log /tmp/servo.log /tmp/relay.log"
log "disturbance test: ./scripts/run.sh cable_qs_disturb   (from the host)"
log "relay recovery events (should stay rare): grep RECOVERY /tmp/relay.log"
log "monitoring /cable/lyapunov [V, dV, ||shape_err||, decrease_flag]"
log "Ctrl-C detaches (stack keeps running); './scripts/run.sh down' stops everything"
exec ros2 topic echo /cable/lyapunov
