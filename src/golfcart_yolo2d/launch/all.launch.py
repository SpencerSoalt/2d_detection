from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import FrontendLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    foxglove_address = LaunchConfiguration("foxglove_address")
    foxglove_port = LaunchConfiguration("foxglove_port")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # foxglove_bridge is XML → FrontendLaunchDescriptionSource is correct
    foxglove_bridge_launch = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("foxglove_bridge"),
                "launch",
                "foxglove_bridge_launch.xml",
            ])
        ),
        launch_arguments={
            "address": foxglove_address,
            "port": foxglove_port,
        }.items(),
    )

    republish_left = Node(
        package="image_transport",
        executable="republish",
        name="republish_front_left",
        output="screen",
        emulate_tty=True,
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "compressed", "raw",
            "--ros-args",
            "-r", "in/compressed:=/camera4/image_raw/compressed",
            "-r", "out:=/front_left/image_raw",
        ],
    )

    republish_right = Node(
        package="image_transport",
        executable="republish",
        name="republish_front_right",
        output="screen",
        emulate_tty=True,
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "compressed", "raw",
            "--ros-args",
            "-r", "in/compressed:=/camera1/image_raw/compressed",
            "-r", "out:=/front_right/image_raw",
        ],
    )

    # YOLO launch is Python → must use PythonLaunchDescriptionSource
    yolo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("golfcart_yolo2d"),
                "launch",
                "yolo12_2d_detector.launch.py",
            ])
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("foxglove_address", default_value="0.0.0.0"),
        DeclareLaunchArgument("foxglove_port", default_value="8765"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),

        foxglove_bridge_launch,
        republish_left,
        republish_right,
        yolo_launch,
    ])
