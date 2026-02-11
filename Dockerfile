FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# ---- Base OS / ROS2 apt setup deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Locale (ROS tools can be picky)
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ---- Install ROS 2 Humble (Ubuntu 22.04) ----
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

# ---- System deps (ROS + tools + your packages) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-dev-tools \
    ros-humble-ros-base \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    python3-opencv \
    tmux \
    vim \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-foxglove-bridge \
    libgl1 \
    libglib2.0-0 \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Set DDS/RMW defaults
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ---- Python deps ----
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel && \
    # CUDA 12.1 wheels for torch/torchvision
    python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    # TensorRT python wheels (kept as you had it)
    python3 -m pip install --no-cache-dir tensorrt-cu12 && \
    python3 -m pip install --no-cache-dir ultralytics && \
    # Prefer system OpenCV (python3-opencv) and avoid pip OpenCV conflicts
    python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true && \
    # Pin numpy to avoid ABI surprises
    python3 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# ---- ROS workspace ----
ENV WS=/ws
WORKDIR ${WS}

# Copy your package into the workspace
# (Assumes your local ./src directory is the package contents; keep as your original)
COPY src ${WS}/src/

# rosdep (safe even if some keys already satisfied)
RUN rosdep init 2>/dev/null || true && \
    rosdep update && \
    source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y

# Build
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# Optional: pre-download YOLO weights at build time (avoids first-run download delay)
ARG YOLO_MODEL=yolo12n.pt
RUN python3 -c "from ultralytics import YOLO; YOLO('${YOLO_MODEL}')"

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]


# FROM ros:humble-ros-base

# SHELL ["/bin/bash", "-c"]

# # ---- System deps ----
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3-pip \
#     python3-colcon-common-extensions \
#     python3-rosdep \
#     ros-dev-tools \
#     ros-humble-vision-msgs \
#     ros-humble-cv-bridge \
#     python3-opencv \
#     tmux \
#     vim \
#     ros-humble-rosbag2 \
#     ros-humble-rosbag2-storage-mcap \
#     ros-humble-image-transport \
#     ros-humble-compressed-image-transport \
#     ros-humble-foxglove-bridge \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     ros-humble-rmw-cyclonedds-cpp \
#     && rm -rf /var/lib/apt/lists/* 

# # Set DDS/RMW defaults
# ENV ROS_DOMAIN_ID=0
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# # ---- Python deps ----
# RUN python3 -m pip install --no-cache-dir --upgrade pip wheel && \
# # Cuda 11.8 GPU version of torch (uncomment to enable GPU support)
#     python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
#     # python3 -m pip install --no-cache-dir nvidia-tensorrt && \
#     python3 -m pip install --no-cache-dir tensorrt-cu12
#     python3 -m pip install --no-cache-dir ultralytics && \
#     python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true && \
#     python3 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# # ---- ROS workspace ----
# ENV WS=/ws
# WORKDIR ${WS}

# # Copy your package into the workspace
# COPY src ${WS}/src/detections_2d

# # rosdep (safe even if some keys already satisfied)
# RUN rosdep init 2>/dev/null || true && rosdep update && \
#     rosdep install --from-paths src --ignore-src -r -y

# # Build
# RUN source /opt/ros/humble/setup.bash && \
#     colcon build --symlink-install

# # Optional: pre-download YOLO weights at build time (avoids first-run download delay)
# # If you prefer not to bake weights into the image, delete these lines.
# ARG YOLO_MODEL=yolo12n.pt
# RUN python3 -c "from ultralytics import YOLO; YOLO('${YOLO_MODEL}')"

# # Entrypoint
# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh

# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["bash"]
