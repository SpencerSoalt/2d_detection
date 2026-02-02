import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import torch
from ultralytics import YOLO
import cv2
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class YOLOv12BatchDetectionNode(Node):
    def __init__(self):
        super().__init__('yolov12_batch_detection_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'yolov12n.pt')
        self.declare_parameter('camera1_topic', '/camera1/image_raw')
        self.declare_parameter('camera2_topic', '/camera2/image_raw')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('classes_to_detect', ['person', 'car'])
        self.declare_parameter('img_size', 320)
        self.declare_parameter('use_openvino', True)
        self.declare_parameter('num_threads', 8)
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        camera1_topic = self.get_parameter('camera1_topic').value
        camera2_topic = self.get_parameter('camera2_topic').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        device = self.get_parameter('device').value
        classes_to_detect = self.get_parameter('classes_to_detect').value
        self.img_size = self.get_parameter('img_size').value
        use_openvino = self.get_parameter('use_openvino').value
        num_threads = self.get_parameter('num_threads').value
        
        # Set thread limits BEFORE loading model
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        
        # PyTorch thread limits
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        
        self.get_logger().info(f'Limiting inference to {num_threads} CPU threads')
        
        # Initialize YOLO model with OpenVINO
        self.get_logger().info(f'Loading YOLOv12 model from {model_path}')
        
        if use_openvino:
            try:
                # Try to load existing OpenVINO model
                openvino_path = model_path.replace('.pt', '_openvino_model')
                self.get_logger().info(f'Attempting to load OpenVINO model from {openvino_path}')
                self.model = YOLO(openvino_path)
                self.get_logger().info('✓ Loaded existing OpenVINO model')
            except:
                self.get_logger().info('OpenVINO model not found, exporting from PyTorch...')
                temp_model = YOLO(model_path)
                # Export to OpenVINO format with batch size 1
                temp_model.export(format='openvino', imgsz=self.img_size, half=False, batch=1)
                openvino_path = model_path.replace('.pt', '_openvino_model')
                self.model = YOLO(openvino_path)
                self.get_logger().info(f'✓ Exported and loaded OpenVINO model: {openvino_path}')
        else:
            # Fallback to PyTorch with optimizations
            self.model = YOLO(model_path)
            torch.backends.mkldnn.enabled = True
            self.get_logger().info('Using PyTorch model with CPU optimizations')
        
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
        
        # Image buffers
        self.img1 = None
        self.img2 = None
        self.img1_header = None
        self.img2_header = None
        
        # Processing flags to avoid duplicate processing
        self.img1_processed = True
        self.img2_processed = True
                
        # Best-effort QoS (good for high-rate camera streams)
        image_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
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
        
        # Publishers for visualization (optional)
        self.vis_pub1 = self.create_publisher(
            Image, '/camera1/detections_image', 10)
        self.vis_pub2 = self.create_publisher(
            Image, '/camera2/detections_image', 10)
        
        self.get_logger().info('YOLOv12 detection node initialized')
    
    def camera1_callback(self, msg):
        self.img1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img1_header = msg.header
        self.img1_processed = False
        self.process_images()
    
    def camera2_callback(self, msg):
        self.img2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.img2_header = msg.header
        self.img2_processed = False
        self.process_images()
    
    def process_images(self):
        # Process camera 1 if new image available
        if self.img1 is not None and not self.img1_processed:
            results = self.model(
                self.img1,
                conf=self.conf_threshold,
                classes=self.class_filter,
                verbose=False,
                imgsz=self.img_size,
                half=False,
                max_det=10
            )
            self.publish_detections(results[0], self.pub1, self.img1_header)
            self.publish_visualization(results[0], self.img1, self.vis_pub1, self.img1_header)
            self.img1_processed = True
        
        # Process camera 2 if new image available
        if self.img2 is not None and not self.img2_processed:
            results = self.model(
                self.img2,
                conf=self.conf_threshold,
                classes=self.class_filter,
                verbose=False,
                imgsz=self.img_size,
                half=False,
                max_det=10
            )
            self.publish_detections(results[0], self.pub2, self.img2_header)
            self.publish_visualization(results[0], self.img2, self.vis_pub2, self.img2_header)
            self.img2_processed = True
    
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
        # Draw detections on image
        annotated_img = result.plot()
        
        # Convert back to ROS message
        vis_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        vis_msg.header = header
        publisher.publish(vis_msg)

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
