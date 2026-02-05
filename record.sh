#!/bin/bash
source /opt/ros/humble/setup.bash

# Record with MCAP format 
ros2 bag record \
    --storage mcap \
    --max-bag-size 5000000000 \
    --max-cache-size 8000000000 \
    /camera1/camera_info \
    /camera2/camera_info \
    /camera1/image_raw/compressed \
    /camera2/image_raw/compressed \
    /fix \
    /heading \
    /imu/mag \
    /ins/imu/data \
    /pacmod/accel_rpt \
    /pacmod/brake_rpt \
    /pacmod/global_rpt \
    /pacmod/shift_rpt \
    /pacmod/steering_rpt \
    /pacmod/vehicle_speed_rpt \
    /vel \
    /velodyne_points \
    /tf_static \
    /tf