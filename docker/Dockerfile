# ============================================================================
# UR5 TS-IBVS ROS2 Jazzy Container
# ============================================================================
# Purpose: UR5 visual servoing with Gazebo Harmonic simulation
# Base: Ubuntu 24.04 + ROS2 Jazzy Jalisco (desktop-full)
# DDS: FastDDS (rmw_fastrtps_cpp)
#
# Architecture (following FETCH pattern):
#   Camera → [TS-IBVS Node] → TwistStamped → [MoveIt Servo] → JointTrajectory → [UR5]
#
# All UR and RealSense packages installed via apt (official releases).
# ur_simulation_gz cloned from source (not yet released to apt for Jazzy).
# ============================================================================

FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================
# STAGE 1: System Dependencies
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Python development
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    # Libraries
    libeigen3-dev \
    libopencv-dev \
    libvisp-dev \
    libvisp-doc \
    libyaml-cpp-dev \
    # GPU / OpenGL rendering for Gazebo
    mesa-utils \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# STAGE 2: ROS2 Package Dependencies (all via apt — official releases)
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # FastDDS (DDS middleware)
    ros-jazzy-rmw-fastrtps-cpp \
    # Vision
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-image-transport-plugins \
    ros-jazzy-vision-opencv \
    ros-jazzy-visp \
    # UR Robot packages (official — same interface as real UR5)
    ros-jazzy-ur-robot-driver \
    ros-jazzy-ur-description \
    ros-jazzy-ur-moveit-config \
    ros-jazzy-ur-controllers \
    ros-jazzy-ur-client-library \
    ros-jazzy-ur-dashboard-msgs \
    ros-jazzy-ur-msgs \
    # Intel RealSense D435 (official packages)
    ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-description \
    ros-jazzy-realsense2-camera-msgs \
    # Gazebo Harmonic (gz-sim8)
    ros-jazzy-gz-sim-vendor \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image \
    ros-jazzy-ros-gz-interfaces \
    ros-jazzy-ros-gz-sim-demos \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-gz-ros2-control-demos \
    # ros2_control
    ros-jazzy-controller-manager \
    ros-jazzy-hardware-interface \
    ros-jazzy-joint-trajectory-controller \
    ros-jazzy-joint-limits \
    ros-jazzy-position-controllers \
    ros-jazzy-velocity-controllers \
    ros-jazzy-effort-controllers \
    ros-jazzy-forward-command-controller \
    ros-jazzy-gripper-controllers \
    ros-jazzy-force-torque-sensor-broadcaster \
    ros-jazzy-control-toolbox \
    # MoveIt2
    ros-jazzy-moveit \
    ros-jazzy-moveit-servo \
    ros-jazzy-moveit-visual-tools \
    ros-jazzy-moveit-ros-perception \
    # PlotJuggler
    ros-jazzy-plotjuggler-ros \
    # Core tools
    ros-jazzy-rqt-common-plugins \
    ros-jazzy-rviz2 \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-tools \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# STAGE 3: DDS Middleware Configuration
# ============================================================================

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ============================================================================
# STAGE 4: GPU Rendering for Gazebo (Ogre2)
# ============================================================================
# Requires nvidia-container-toolkit on the host + runtime: nvidia in compose.
# Mesa kept as fallback; NVIDIA EGL ICD added so Ogre2 Sensors system can
# create a headless EGL context for camera rendering.

# Mesa (fallback / Intel iGPU support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    llvm libllvm18 \
    libgl1-mesa-dri \
    mesa-va-drivers \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA EGL ICD — enables Ogre2 headless EGL rendering via NVIDIA GPU.
# The nvidia-container-toolkit mounts libEGL_nvidia.so.0 at runtime.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ============================================================================
# STAGE 5: Python Dependencies (LMI solver)
# ============================================================================

RUN printf "numpy==1.26.4\nscipy==1.11.4\n" > /tmp/constraints.txt && \
    pip3 install --no-cache-dir --break-system-packages \
    -c /tmp/constraints.txt \
    cvxpy && \
    rm /tmp/constraints.txt

# ============================================================================
# STAGE 6: SOFA v25.12.00 (soft-body simulation framework)
# ============================================================================
# Official Linux-Python_3.12 build — matches Ubuntu 24.04 / ROS2 Jazzy's
# Python 3.12, so SofaPython3 and rclpy run in the SAME interpreter. A SOFA
# Python scene controller can therefore publish/subscribe ROS2 topics that
# ros_gz_bridge connects to Gazebo.
#
# IMPORTANT: SOFA bundles its own Qt 5.12 in /opt/sofa/lib. It must NEVER be
# put on the global LD_LIBRARY_PATH or it breaks RViz/rqt/Gazebo GUIs
# (Qt 5.15). SOFA binaries/libs carry $ORIGIN-based RUNPATHs and self-resolve;
# the /usr/local/bin/runSofa wrapper scopes LD_LIBRARY_PATH to SOFA only.

RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    python3-dev \
    libopengl0 \
    libglu1-mesa \
    libxkbcommon-x11-0 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
    libxcb-xfixes0 libxcb-util1 libxcb-glx0 \
    libx11-xcb1 libsm6 libice6 \
    libgtk-3-0t64 \
    && rm -rf /var/lib/apt/lists/*

# pybind11 runtime module for SofaPython3 (numpy/scipy already installed)
RUN pip3 install --no-cache-dir --break-system-packages pybind11==2.12.0

ARG SOFA_URL=https://github.com/sofa-framework/sofa/releases/download/v25.12.00/SOFA_v25.12.00_Linux-Python_3.12.zip
ARG SOFA_SHA256=76be4a5d89656a7e8c89bc45a30665aab2a52ba33e77e2a59280b735a9288574

# Download in its own layer so it stays cached across rebuilds.
RUN curl -L -o /tmp/sofa.zip "$SOFA_URL" && \
    echo "${SOFA_SHA256}  /tmp/sofa.zip" | sha256sum -c -

# Extract: the archive contains a single top-level dir.
RUN mkdir -p /tmp/sofa-extract /opt/sofa && \
    unzip -q /tmp/sofa.zip -d /tmp/sofa-extract && \
    SOFA_DIR="$(find /tmp/sofa-extract -mindepth 1 -maxdepth 1 -type d | head -n1)" && \
    if [ -z "$SOFA_DIR" ] || [ ! -e "$SOFA_DIR/bin/runSofa" ]; then \
        echo "ERROR: SOFA root not found in archive" >&2; ls -la /tmp/sofa-extract; exit 1; \
    fi && \
    cp -a "$SOFA_DIR"/. /opt/sofa/ && \
    rm -rf /tmp/sofa.zip /tmp/sofa-extract && \
    chmod +x /opt/sofa/bin/runSofa*

ENV SOFA_ROOT=/opt/sofa

# SofaPython3 bindings + plugin Python libs (stlib3, softrobots, mor, ...).
# ROS setup.bash prepends to PYTHONPATH at runtime, so both coexist.
ENV PYTHONPATH=/opt/sofa/plugins/SofaPython3/lib/python3/site-packages:/opt/sofa/plugins/STLIB/lib/python3/site-packages:/opt/sofa/plugins/SoftRobots/lib/python3/site-packages:/opt/sofa/plugins/SoftRobots.Inverse/lib/python3/site-packages:/opt/sofa/plugins/ModelOrderReduction/lib/python3/site-packages:/opt/sofa/plugins/SofaGLFW/lib/python3/site-packages:/opt/sofa/plugins/SofaImGui/lib/python3/site-packages:/opt/sofa/plugins/SofaImplicitField/lib/python3/site-packages:/opt/sofa/plugins/Sofa.Qt/lib/python3/site-packages:/opt/sofa/plugins/SofaViscoElastic/lib/python3/site-packages

# Scoped runSofa launcher: SOFA's private Qt/libs affect only this process.
# Auto-loads SofaPython3 when a Python scene is passed.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'export LD_LIBRARY_PATH=/opt/sofa/lib:/opt/sofa/plugins${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' \
    'PY_PLUGIN=()' \
    'for arg in "$@"; do' \
    '  case "${arg,,}" in' \
    '    *.py|*.py3|*.pyscn|*.py3scn) PY_PLUGIN=(-l SofaPython3); break;;' \
    '  esac' \
    'done' \
    'exec /opt/sofa/bin/runSofa "${PY_PLUGIN[@]}" "$@"' \
    > /usr/local/bin/runSofa && chmod +x /usr/local/bin/runSofa

# --- Cosserat plugin (cable physics), pinned release for SOFA v25.12 -------
# The binary links against dev-suffix sonames (.so.25.12.99) while the SOFA
# release ships .so.25.12.00 -> compatibility symlinks next to the plugin lib.
ARG COSSERAT_URL=https://github.com/SofaDefrost/Cosserat/releases/download/release-v25.12/Cosserat_v25.12_for-SOFA-v25.12_Linux.zip
ARG COSSERAT_SHA256=e6d9476e6543f0cff7a0819ddbbbf1b51401cee1f4147346d89a9ab02f19bb8c
RUN curl -L -o /tmp/cosserat.zip "$COSSERAT_URL" && \
    echo "${COSSERAT_SHA256}  /tmp/cosserat.zip" | sha256sum -c - && \
    mkdir -p /tmp/cosserat-extract && \
    unzip -q /tmp/cosserat.zip -d /tmp/cosserat-extract && \
    SRC="$(find /tmp/cosserat-extract -mindepth 1 -maxdepth 1 -type d | head -n1)" && \
    mkdir -p /opt/sofa/plugins/Cosserat && \
    cp -a "$SRC"/. /opt/sofa/plugins/Cosserat/ && \
    rm -rf /tmp/cosserat.zip /tmp/cosserat-extract && \
    cd /opt/sofa/plugins/Cosserat/lib && \
    for f in /opt/sofa/lib/libSofa*.so.25.12.00; do \
        b=$(basename "$f"); ln -sf "$f" "${b%25.12.00}25.12.99"; \
    done

# --- Optimus (stochastic parameter identification) ported to SOFA v25.12 -----
# Upstream Optimus (sofa-framework/Optimus @ cd41bf17) only builds against SOFA
# 21.12. third_party/Optimus is the minimal-core port to 25.12 (modular CMake
# targets, registerObjects entry point, moved headers, ROUKF/observation fixes;
# see docs/optimus_port.md). The same stage rebuilds Cosserat FROM SOURCE at
# the exact commit of the binary release above (f64e029) with
# third_party/cosserat-patches: BeamHookeLawForceField must track EI/GI as
# internal data, otherwise the stiffness cache is never rebuilt when the
# estimator writes a candidate value and the parameter gain is identically 0.
# Gates: `ros2 run cable_identification optimus_smoke_test` (factory) and
# `ros2 run cable_identification optimus_recovery_test` (EI/GJ recovery).
# scripts/install_optimus.sh is the single code path (also used to repair a
# recreated container), so the image and the live install cannot drift.
ARG WITH_OPTIMUS=true
RUN apt-get update && apt-get install -y --no-install-recommends \
        libeigen3-dev libblas-dev liblapack-dev libboost-dev patch && \
    rm -rf /var/lib/apt/lists/*
COPY third_party/ /ros2_ws/third_party/
COPY scripts/install_optimus.sh /ros2_ws/scripts/install_optimus.sh
RUN if [ "$WITH_OPTIMUS" = "true" ]; then \
        JOBS=2 bash /ros2_ws/scripts/install_optimus.sh && \
        rm -rf /opt/build/cosserat /opt/build/cosserat-src /opt/build/optimus; \
    else \
        echo "Optimus skipped (build with --build-arg WITH_OPTIMUS=true)"; \
    fi
ENV OPTIMUS_ROOT=/opt/sofa/plugins/Optimus

# ============================================================================
# STAGE 7: Workspace Setup
# ============================================================================

ENV WS=/ros2_ws
RUN mkdir -p ${WS}/src
WORKDIR ${WS}

# Clone ur_simulation_gz from source (not yet released to apt for Jazzy).
# Provides the gz-sim ros2_control hardware interface for UR robots.
RUN git clone -b ros2 \
    https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git \
    ${WS}/src/ur_simulation_gz

# Note: The realsense_gazebo_plugin (gitioc.upc.edu) requires Gazebo Classic,
# which is incompatible with Gazebo Harmonic (gz-sim8). Instead, we use native
# gz-sim <sensor> tags in the URDF + ros_gz_bridge for camera simulation.
# For the real robot, use ros-jazzy-realsense2-camera (already installed above).

# Copy source packages
COPY src/ ${WS}/src/

# Install missing rosdep dependencies
RUN apt-get update && \
    rosdep update --rosdistro=jazzy && \
    rosdep install --ignore-src --from-paths src -y || true && \
    rm -rf /var/lib/apt/lists/*

# Build workspace (need bash for 'source')
# Sequential with capped compiler jobs: the default 16-way parallelism OOMs on
# an 8 GB machine and kills the build half way through.
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/jazzy/setup.bash && \
    MAKEFLAGS=-j2 colcon build --executor sequential --parallel-workers 1 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# ============================================================================
# STAGE 8: ROS2 Environment Setup
# ============================================================================

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source ${WS}/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> /root/.bashrc

# ============================================================================
# STAGE 9: Entrypoint
# ============================================================================

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
