---
name: cable-live-stack
description: "Start, stop, inspect or debug the live Gazebo + SOFA + MoveIt Servo cable stack in the Docker container: cable_sim, cable_closed_loop, cable_qs_test demo, attach the cable, pose the arm, kill zombie nodes, OOM/Gazebo crashes, runSofa segfaults, stale installs after container recreation, missing segments, 'stale; holding', servo 'Command type has not been set', ros2/gz 'command not found'. Use for any live cable experiment or sim troubleshooting."
argument-hint: "what you want to do with the live stack (start demo, stop, diagnose <symptom>)"
---
# Live cable stack (Gazebo + SOFA + Servo)

All commands run from the host repo root via `./scripts/run.sh`; everything
else goes through `docker compose exec ur5e_ts_ibvs bash -c '...'` and must
`source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash` first.
Reference: [docs/cable_ts_status_and_diagnosis.md](../../../docs/cable_ts_status_and_diagnosis.md) §6.

## Start
| Goal | Command |
|---|---|
| Plant only (Gazebo + runSofa GUI, no control) | `./scripts/run.sh cable_sim [sofa_gui:=false]` |
| Full loop, own control of args | `./scripts/run.sh cable_closed_loop observation_source:=camera enable_control:=true ...` |
| Certified one-shot demo (teardown, sim, pose, attach, servo, latch, relay, monitor) | `./scripts/run.sh cable_qs_test [pose]` (default pose `cable_high`; if attach hangs retry `cable_low`) |
| Disturbance pulse on a running demo | `./scripts/run.sh cable_qs_disturb` |
| Move arm | `./scripts/run.sh pose <name>` (names in [move_to_view_pose.py](../../../src/ibvs_control/scripts/move_to_view_pose.py)) |
| Attach cable | `ros2 service call /cable/attach std_srvs/srv/Trigger`, then `/cable/grasp_state` field `state` == 2 |
| Identification only (truth + estimator, no control) | `./scripts/run.sh cable_identify_sim`; headless gate without Gazebo: `./scripts/run.sh cable_optimus_pipeline [--duration 60]` (refuses to start over a live stack) |

Check `free -g` first: Gazebo OOM-dies silently below ~3 GB available (no gz
processes, no log line). Only [qs_live_test.sh](../../../scripts/cable/qs_live_test.sh)
is the verified sequence; copy its ordering rather than inventing a new one.
Every `run.sh` sim command and every `ros2 launch` blocks forever: run it in a
background terminal or `docker exec -d`, then read `/tmp/*.log`; never wait on it
in the foreground.

## Ordering rules that bit before
- Servo must be started **after** posing: it streams hold trajectories at ~29 Hz to
  `/joint_trajectory_controller/joint_trajectory` and pre-empts `move_to_view_pose`.
  The demo passes `use_servo:=false` and starts `ros2 launch ibvs_control
  servo.launch.py ur_type:=ur5e use_sim_time:=true` itself.
- After Servo is up, switch to twist mode or nothing moves ("Command type has not
  been set"): `ros2 service call /servo_node/switch_command_type
  moveit_msgs/srv/ServoCommandType "{command_type: 1}"`.
- Publishing to `/joint_trajectory_controller/joint_trajectory` from a script:
  wait for `get_subscription_count() > 0` or the message is lost.
- Detached processes: `docker exec -d ...`; `nohup ... &` dies with the exec.
- `pkill -f` from inside `bash -c`: use bracket patterns (`"[t]wist_relay"`) and
  keep the pkill and any later launch of the same name in separate execs.
- `ros2 run` children (SOFA python nodes) survive SIGTERM to the wrapper and
  sometimes `pkill -f`; confirm with `pgrep -fa "[c]able_estimator_node|[c]able_sofa_node|[s]ynthetic_marker_node"`
  and `kill -9 <pid>` the survivors. Two estimators on the same topics produce an
  alternating `/cable/parameter_estimate` trace, not an error.

## Stop
`./scripts/run.sh down` stops everything (Cosserat and Optimus live in the
container overlay unless the image was rebuilt with the Dockerfile Optimus stage;
the next `cable_*` command re-installs them, ~10 min for the source builds). For a
hot restart inside the running container, reuse the
teardown loop at the top of [qs_live_test.sh](../../../scripts/cable/qs_live_test.sh):
it lists every process that must die (`cable_*`, `servo_node`,
`ros2_control_node`, `robot_state_publisher`, `static_transform_publisher`,
`parameter_bridge`, `gz sim`). Wait ~3 s, then `pkill -9 -f "[g]z sim"`.

## Diagnose
| Symptom | Cause / check |
|---|---|
| `ros2`/`gz`: command not found | exec shell missing the two `source` lines above |
| `runSofa ... process has died [exit code -11]` right after launch | After container recreation, first rebuild `cable_identification` + `sofa_ros2_adapter`: stale installed Python copies can restore a fixed crash (see [container model](../../../AGENTS.md#container-model-read-before-running-anything)). If it persists, inspect `/tmp/cable_sim.log` for pre-init `reinit()` or mapped-state constraints; follow the [SOFA instructions](../../instructions/sofa-cosserat.instructions.md) and run `cable_forward_test` before relaunching. |
| Cable does not attach / follow the gripper | `/cable/grasp_state`: `in_range` false = pose too far (use `cable_high`/`cable_low`, tip within `distance_to_tip_m`); `state` stuck at 1 = truth node dead (`pgrep -f cable_sofa_node`), or `sofa_gui:=true` runSofa crashed (above); `state` 2 but no motion = Servo not in twist mode |
| Cable invisible or moves "by cuts" | `cable_visual_node` dead (partner `set_pose_vector` died): restart `ros2 run gazebo_cable_visual cable_visual_node -p world:=ibvs_world -p world_frame:=world -p frames_topic:=/cable/truth/frames -p use_sim_time:=true` |
| Rendered cable has gaps, truth continuous | create/remove race; `healMissingSegments` respawns every ~5 s, wait |
| Metric flaps ~1 Hz healthy/coiled | zombie publisher: `ros2 topic info -v /cable/truth/frames` must show 1 publisher, `pgrep -f cable_sofa_node` exactly 1 |
| Controller "stale; holding", reducer `valid=False` | shape off the narrow-basis manifold (hanging pre-grasp cable, out-of-envelope) or DLO detector lost length ("detected chain 0.42 m expected 0.7": arm occludes overview camera) |
| Detector picks the arm | cable must be blue; HSV `[100,120,60]-[140,255,255]` set in the closed-loop launch |
| Supervisor FAULT | recovers after `recover_time_s` (3 s) of healthy state; it adopts an external `CableShapeTarget` (name+coords change). States INITIALIZE/APPROACH/GRASP/SHAPE/HOLD/FAULT on `/cable/supervisor_state` |
| Estimator rejects most observations, EI stalls | CPU starvation: `/cable/truth/solver_stats[1]` (dropped steps) rising; kill stray `ros2 topic echo`/browsers, do not touch `innovation_gate_sigma` |
| Gripper walks out of premise box | known: DLO bias in weak modes vs zero g-feedback columns (docs §7); bias anchor + relay backstop, not a bug to "fix" in the gains |
| Cable snaps/coils on attach | scene rule violation; see the `sofa-cosserat` instructions, not a tuning issue |

Logs: `/tmp/cable_sim.log`, `/tmp/servo.log`, `/tmp/relay.log` in the
container (`grep RECOVERY /tmp/relay.log`), CSVs in `~/ibvs_logs` on the host.
Topics worth echoing: `/cable/lyapunov` `[V, dV, ||err||, decrease]`,
`/cable/reduced_state`, `/cable/ts_weights` (`[w0..w3, cmd_x, cmd_y, |cmd|, est_fresh]`:
fields 5-6 are the command, not premises), `/cable/grasp_state`.

## Knobs that exist (do not re-implement them)
| Where | Parameters |
|---|---|
| `cable_state_reducer_node` | `max_reconstruction_rmse_m` 0.05, `max_modal_jump_m` 0.05 + `jump_confirm_samples` 3, `modal_smoothing_alpha` 0.35 |
| `cable_ts_controller_node` | `enable_bias_anchor`, `bias_anchor_alpha` 0.05, `bias_anchor_max_m` 0.05 (manifold-bias observer, docs audit 9.1) |
| `cable_supervisor_node` | `state_timeout_s` 3, `recover_time_s` 3, `shape_tolerance_m` 0.01, `hold_time_s` 2, `auto_start` |
| `cable_dlo_detector_node` | `cable_hsv_lower/upper`, `plane_normal`, `plane_offset_m`, `marker_s_over_l` |
| `cable_closed_loop_sim.launch.py` | `observation_source` synthetic\|camera, `enable_optimus`, `enable_control`, `marker_noise_std_m`, `marker_dropout`, `marker_delay_s`, `*_file` artifact paths |
| [twist_relay.py](../../../scripts/cable/twist_relay.py) | premise box `FORE_LO/HI` [0.06, 0.11], `BEAR_LIM` 0.25, `RECOVER_SPEED` (envelope steering, not control) |
| [approach3.py](../../../scripts/cable/approach3.py) | `Q_WINDOW_S` 10, `Q_PTP` 0.004 (stationary latch; a premature latch chases a non-equilibrium) |
