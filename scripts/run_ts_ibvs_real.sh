#!/bin/bash
# Real-robot wrapper (parity with the ROS1 project script of the same name).
# Forwards extra args to ibvs.launch.py, e.g.:
#   ./run_ts_ibvs_real.sh cube_size_x:=0.075 cube_size_y:=0.060 mount_z:=0.05
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run.sh" real ts_lmi_d robot_ip:=192.168.1.133 "$@"
