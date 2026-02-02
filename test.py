import threading
from rclpy.executors import MultiThreadedExecutor

class YOLOv12BatchDetectionNode(Node):
    def __init__(self):
        ...
        self.lock = threading.Lock()
        self.img1 = self.img2 = None
        self.img1_header = self.img2_header = None
        self.busy = False

        # Use depth=1 for "latest only"
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub1 = self.create_subscription(Image, camera1_topic, self.camera1_callback, image_qos)
        self.sub2 = self.create_subscription(Image, camera2_topic, self.camera2_callback, image_qos)

        # Run inference at max possible rate without blocking subscriptions
        self.timer = self.create_timer(0.0, self.process_batch)

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

            # Optional: throttle visualization hard or disable it
            # self.publish_visualization(...)
        finally:
            with self.lock:
                self.busy = False

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv12BatchDetectionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
