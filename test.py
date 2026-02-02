#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import threading
import torch
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class YOLOv12BatchDetectionNode(Node):
    def __init__(self):
        super().__init__('yolov12_batch_detection_node')

        # -------------------------
        # Parameters (UNCHANGED)
        # -------------------------
        self.declare_parameter('model_path', 'yolov12n.pt')
        self.declare_parameter('camera1_topic', '/camera1/image_raw')
        self.declare_parameter('camera2_topic', '/camera2/image_raw')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('classes_to_detect', ['person', 'car'])

        model_path = self.get_parameter('model_path').value
        camera1_topic = self.get_parameter('camera1_topic').value
        camera2_topic = self.get_parameter('camera2_topic').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        device = self.get_parameter('device').value
        classes_to_detect = self.get_parameter('classes_to_detect').value

        self.get_logger().info(f'Loading YOLOv12 model from {model_path}')
        self.model = YOLO(model_path)
        self.model.to(device)

        self.class_names = self.model.names
        self.class_filter = []
        for class_name in classes_to_detect:
            for class_id, name in self.class_names.items():
                if name.lower() == class_name.lower():
                    self.class_filter.append(class_id)
                    break

        self.get_logger().info(f'Filtering for classes: {classes_to_detect}')
        self.get_logger().info(f'Class IDs: {self.class_filter}')

        # -------------------------
        # Runtime state
        # -------------------------
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.busy = False

        self.img1 = None
        self.img2 = None
        self.img1_header = None
        self.img2_header = None

        # QoS: "latest frame wins"
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # was 5
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers (callbacks now only store frames)
        self.sub1 = self.create_subscription(Image, camera1_topic, self.camera1_callback, image_qos)
        self.sub2 = self.create_subscription(Image, camera2_topic, self.camera2_callback, image_qos)

        # Publishers
        self.pub1 = self.create_publisher(Detection2DArray, '/camera1/detections', 10)
        self.pub2 = self.create_publisher(Detection2DArray, '/camera2/detections', 10)

        self.vis_pub1 = self.create_publisher(Image, '/camera1/detections_image', 10)
        self.vis_pub2 = self.create_publisher(Image, '/camera2/detections_image', 10)

        # Timer runs inference without blocking subscriptions
        self.timer = self.create_timer(0.0, self.process_batch)

        self.get_logger().info('YOLOv12 batch detection node initialized')

    def camera1_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.lock:
            self.img1 = img
            self.img1_header = msg.header

    def camera2_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.lock:
            self.img2 = img
            self.img2_header = msg.header

    def process_batch(self):
        # grab latest pair (don’t block callbacks too long)
        with self.lock:
            if self.busy or self.img1 is None or self.img2 is None:
                return
            self.busy = True
            img1, img2 = self.img1, self.img2
            h1, h2 = self.img1_header, self.img2_header

        try:
            results = self.model([img1, img2], conf=self.conf_threshold, classes=self.class_filter, verbose=False)

            self.publish_detections(results[0], self.pub1, h1)
            self.publish_detections(results[1], self.pub2, h2)

            # Strongly consider throttling visualization
            # self.publish_visualization(results[0], img1, self.vis_pub1, h1)
            # self.publish_visualization(results[1], img2, self.vis_pub2, h2)

        finally:
            with self.lock:
                self.busy = False

    def publish_detections(self, result, publisher, header):
        detection_array = Detection2DArray()
        detection_array.header = header

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            publisher.publish(detection_array)
            return

        # Move tensors to CPU ONCE (avoid per-box .cpu())
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        conf = boxes.conf.detach().cpu().numpy()

        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            detection = Detection2D()
            detection.header = header

            detection.bbox.center.position.x = float((x1 + x2) / 2.0)
            detection.bbox.center.position.y = float((y1 + y2) / 2.0)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(c))
            hypothesis.hypothesis.score = float(s)
            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        publisher.publish(detection_array)

    def publish_visualization(self, result, img, publisher, header):
        annotated_img = result.plot()
        vis_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        vis_msg.header = header
        publisher.publish(vis_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv12BatchDetectionNode()

    # Multi-threaded executor prevents subscription starvation
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
