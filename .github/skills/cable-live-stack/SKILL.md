---
name: cable-live-stack
description: "Start, stop, inspect or debug the live Gazebo + SOFA + MoveIt Servo cable stack in the Docker container: cable_sim, cable_closed_loop, cable_qs_test demo, attach the cable, pose the arm, kill zombie nodes, OOM/Gazebo crashes, missing segments, 'stale; holding', servo 'Command type has not been set', ros2/gz 'command not found'. Use for any live cable experiment or sim troubleshooting."
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
| Cable invisible or moves "by cuts" | `cable_visual_node` dead (partner `set_pose_vector` died): restart `ros2 run gazebo_cable_visual cable_visual_node -p world:=ibvs_world -p world_frame:=world -p frames_topic:=/cable/truth/frames -p use_sim_time:=true` |
| Rendered cable has gaps, truth continuous | create/remove race; `healMissingSegments` respawns every ~5 s, wait |
| Metric flaps ~1 Hz healthy/coiled | zombie publisher: `ros2 topic info -v /cable/truth/frames` must show 1 publisher, `pgrep -f cable_sofa_node` exactly 1 |
| Controller "stale; holding", reducer `valid=False` | shape off the narrow-basis manifold (hanging pre-grasp cable, out-of-envelope) or DLO detector lost length ("detected chain 0.42 m expected 0.7": arm occludes overview camera) |
| Detector picks the arm | cable must be blue; HSV `[100,120,60]-[140,255,255]` set in the closed-loop launch |
| Supervisor FAULT | recovers after `recover_time_s` (3 s) of healthy state; it adopts an external `CableShapeTarget` (name+coords change) |
| Gripper walks out of premise box | known: DLO bias in weak modes vs zero g-feedback columns (docs §7); bias anchor + relay backstop, not a bug to "fix" in the gains |
| Cable snaps/coils on attach | scene rule violation; see the `sofa-cosserat` instructions, not a tuning issue |

Logs: `/tmp/cable_sim.log`, `/tmp/servo.log`, `/tmp/relay.log` in the
container (`grep RECOVERY /tmp/relay.log`), CSVs in `~/ibvs_logs` on the host.
Topics worth echoing: `/cable/lyapunov` `[V, dV, ||err||, decrease]`,
`/cable/reduced_state`, `/cable/ts_weights` (`[w0..w3, cmd_x, cmd_y, |cmd|, est_fresh]`:
fields 5-6 are the command, not premises), `/cable/grasp_state`.
