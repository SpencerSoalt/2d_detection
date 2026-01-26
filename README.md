docker run -it --rm -p 8765:8765 -v {absolute path to bags}:/ws/bags 2d_detection:humble bash


docker build -t 2d_detection:humble {path to dockerfile dir}

docker run -it --rm --gpus all --name 2d_detec -p 8765:8765 -v {absolute path to bags}:/bags 2d_detection:humble bash

ros2 bag play /bags/run1 --clock -l

ros2 launch golfcart_yolo2d  all.launch.py

ros2 launch golfcart_yolo2d yolo12_2d_detector.launch.py






Decompress: left:
ros2 run image_transport republish compressed raw --ros-args \
-r in/compressed:=/camera4/image_raw/compressed \
-r out:=/front_left/image_raw

right:
ros2 run image_transport republish compressed raw --ros-args \
-r in/compressed:=/camera1/image_raw/compressed \
-r out:=/front_right/image_raw



ros2 launch golfcart_yolo2d yolo12_2d_detector.launch.py 


ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=0.0.0.0 port:=8765



ros2 launch golfcart_yolo2d yolo12_2d_detector.launch.py --ros-args \
  -p model:=yolo12l.pt

