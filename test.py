#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import threading
from queue import Queue, Empty


class YOLOv12BatchDetectionNode(Node):
    def __init__(self):
        super().__init__('yolov12_batch_detection_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'yolov12n.pt')
        self.declare_parameter('camera1_topic', '/camera1/image_raw')
        self.declare_parameter('camera2_topic', '/camera2/image_raw')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('classes_to_detect', ['person', 'car'])
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        camera1_topic = self.get_parameter('camera1_topic').value
        camera2_topic = self.get_parameter('camera2_topic').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        device = self.get_parameter('device').value
        classes_to_detect = self.get_parameter('classes_to_detect').value
        
        # Initialize YOLO model
        self.get_logger().info(f'Loading YOLOv12 model from {model_path}')
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Map class names to class IDs
        self.class_names = self.model.names
        self.class_filter = []
        for class_name in classes_to_detect:
            for class_id, name in self.class_names.items():
                if name.lower() == class_name.lower():
                    self.class_filter.append(class_id)
                    break
        
        self.get_logger().info(f'Filtering for classes: {classes_to_detect}')
        self.get_logger().info(f'Class IDs: {self.class_filter}')
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Image buffers for batch processing (protected by lock)
        self.img1 = None
        self.img2 = None
        self.img1_header = None
        self.img2_header = None
        self.buffer_lock = threading.Lock()
        
        # Queue for passing image pairs to inference thread
        self.inference_queue = Queue(maxsize=1)  # Only keep latest pair
        
        # Start inference thread
        self.running = True
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

        # Best-effort QoS
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Reduced from 5 - only keep latest
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.sub1 = self.create_subscription(
            Image, camera1_topic, self.camera1_callback, image_qos)
        self.sub2 = self.create_subscription(
            Image, camera2_topic, self.camera2_callback, image_qos)
        
        # Publishers for detections
        self.pub1 = self.create_publisher(
            Detection2DArray, '/camera1/detections', 10)
        self.pub2 = self.create_publisher(
            Detection2DArray, '/camera2/detections', 10)
        
        # Publishers for visualization
        self.vis_pub1 = self.create_publisher(
            Image, '/camera1/detections_image', 10)
        self.vis_pub2 = self.create_publisher(
            Image, '/camera2/detections_image', 10)
        
        self.get_logger().info('YOLOv12 batch detection node initialized')
    
    def camera1_callback(self, msg):
        """Fast callback - just store image and try to queue"""
        with self.buffer_lock:
            self.img1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.img1_header = msg.header
            self.try_queue_batch()
    
    def camera2_callback(self, msg):
        """Fast callback - just store image and try to queue"""
        with self.buffer_lock:
            self.img2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.img2_header = msg.header
            self.try_queue_batch()
    
    def try_queue_batch(self):
        """Queue image pair for inference if both available (called with lock held)"""
        if self.img1 is None or self.img2 is None:
            return
        
        # Prepare data to send to inference thread
        batch_data = (
            self.img1.copy(),
            self.img2.copy(),
            self.img1_header,
            self.img2_header
        )
        
        # Clear buffers
        self.img1 = None
        self.img2 = None
        
        # Non-blocking put - if queue is full, clear it and add new
        try:
            # Try to empty the queue first (drop old frame)
            try:
                self.inference_queue.get_nowait()
            except Empty:
                pass
            self.inference_queue.put_nowait(batch_data)
        except:
            pass  # Should not happen with maxsize=1 after clearing
    
    def inference_loop(self):
        """Separate thread for GPU inference - doesn't block callbacks"""
        self.get_logger().info('Inference thread started')
        
        while self.running and rclpy.ok():
            try:
                # Wait for image pair with timeout
                img1, img2, h1, h2 = self.inference_queue.get(timeout=0.1)
                
                # Run batch inference (this is the slow part)
                results = self.model(
                    [img1, img2], 
                    conf=self.conf_threshold, 
                    classes=self.class_filter, 
                    verbose=False
                )
                
                # Publish results (publishers are thread-safe in ROS2)
                self.publish_detections(results[0], self.pub1, h1)
                self.publish_visualization(results[0], img1, self.vis_pub1, h1)
                
                self.publish_detections(results[1], self.pub2, h2)
                self.publish_visualization(results[1], img2, self.vis_pub2, h2)
                
            except Empty:
                continue  # No data available, keep waiting
            except Exception as e:
                self.get_logger().error(f'Inference error: {e}')
    
    def publish_detections(self, result, publisher, header):
        detection_array = Detection2DArray()
        detection_array.header = header
        
        boxes = result.boxes
        for i in range(len(boxes)):
            detection = Detection2D()
            detection.header = header
            
            # Bounding box
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Classification
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(boxes.cls[i].item()))
            hypothesis.hypothesis.score = float(boxes.conf[i].item())
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        publisher.publish(detection_array)
    
    def publish_visualization(self, result, img, publisher, header):
        annotated_img = result.plot()
        vis_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        vis_msg.header = header
        publisher.publish(vis_msg)
    
    def destroy_node(self):
        """Clean shutdown"""
        self.running = False
        self.inference_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv12BatchDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
