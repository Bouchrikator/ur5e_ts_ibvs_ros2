#!/bin/bash
set -e

# Source ROS2 base
source /opt/ros/jazzy/setup.bash

# Source workspace if built
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

# Set FastDDS
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Create XDG_RUNTIME_DIR if needed
if [ -n "$XDG_RUNTIME_DIR" ] && [ ! -d "$XDG_RUNTIME_DIR" ]; then
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 0700 "$XDG_RUNTIME_DIR"
fi

exec "$@"
