FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-dev-tools \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    python3-opencv \
    tmux \
    vim \
    ros-humble-rosbag2 \
    # ros-humble-rosbag2-storage-default-plugins \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/* 
    

# ---- Python deps ----
# ultralytics pulls in torch deps via pip (CPU by default). For GPU, see notes below.
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel && \
    python3 -m pip install --no-cache-dir ultralytics && \
    python3 -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true && \
    python3 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# ---- ROS workspace ----
ENV WS=/ws
WORKDIR ${WS}

# Copy your package into the workspace
COPY src ${WS}/src/golfcart_yolo2d

# rosdep (safe even if some keys already satisfied)
RUN rosdep init 2>/dev/null || true && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# Optional: pre-download YOLO weights at build time (avoids first-run download delay)
# If you prefer not to bake weights into the image, delete these lines.
ARG YOLO_MODEL=yolo12n.pt
RUN python3 -c "from ultralytics import YOLO; YOLO('${YOLO_MODEL}')"

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
