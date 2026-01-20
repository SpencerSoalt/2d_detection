from setuptools import setup
from glob import glob
import os

package_name = "golfcart_yolo2d"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@todo.todo",
    description="YOLOv12 (Ultralytics YOLO12) 2D detector for two frontal cameras; publishes vision_msgs/Detection2DArray.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "yolo12_2d_detector = golfcart_yolo2d.yolo12_2d_detector_node:main",
        ],
    },
)
