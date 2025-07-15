"""
ROS1 Noetic Communication Handler for SLAM Integration
Specialized handler for ROS1 Noetic with custom message types for better navigation integration
"""

import json
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# ROS1 Noetic imports
try:
    import rospy
    from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped, Transform, TransformStamped
    from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
    from std_msgs.msg import Header, String
    from nav_msgs.msg import OccupancyGrid, MapMetaData
    from visualization_msgs.msg import Marker, MarkerArray
    import tf2_ros
    import tf2_geometry_msgs
    from tf2_msgs.msg import TFMessage
    import tf
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("⚠️  ROS1 Noetic not available. Install ROS1 Noetic and required packages.")

# Try to import custom messages (they need to be built first)
try:
    from realtime_detection_msgs.msg import DetectedObject, DetectedObjects
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False
    print("⚠️  Custom messages not available. Build the realtime_detection_msgs package first.")

class ROS1NoeticCommunication:
    """
    Specialized ROS1 Noetic communication handler for navigation systems
    """
    
    def __init__(self, 
                 node_name: str = "realtime_object_detector",
                 objects_topic: str = "/detected_objects",
                 markers_topic: str = "/detection_markers",
                 pointcloud_topic: str = "/detection_pointcloud",
                 camera_info_topic: str = "/camera/camera_info",
                 tf_frame: str = "camera_link",
                 world_frame: str = "map"):
        
        if not ROS_AVAILABLE:
            raise ImportError("ROS1 Noetic not available. Install ROS1 Noetic and required packages.")
        
        self.node_name = node_name
        self.objects_topic = objects_topic
        self.markers_topic = markers_topic
        self.pointcloud_topic = pointcloud_topic
        self.camera_info_topic = camera_info_topic
        self.tf_frame = tf_frame
        self.world_frame = world_frame
        
        # Publishers
        self.objects_publisher = None
        self.markers_publisher = None
        self.pointcloud_publisher = None
        self.camera_info_publisher = None
        
        # TF handling
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_broadcaster = None
        
        # Connection state
        self.connected = False
        self.node_initialized = False
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'objects_published': 0,
            'markers_published': 0,
            'tf_published': 0,
            'start_time': time.time()
        }
        
        print(f"🔧 ROS1 Noetic Communication Handler initialized")
        print(f"📡 Node: {node_name}")
        print(f"🎯 Objects topic: {objects_topic}")
        print(f"📍 Markers topic: {markers_topic}")
        print(f"🌐 TF frame: {tf_frame} -> {world_frame}")
    
    def connect(self) -> bool:
        """Initialize ROS1 Noetic node and publishers"""
        try:
            # Initialize ROS node if not already initialized
            if not self.node_initialized:
                try:
                    rospy.init_node(self.node_name, anonymous=True)
                    self.node_initialized = True
                    print(f"✅ ROS1 node initialized: {self.node_name}")
                except rospy.exceptions.ROSException as e:
                    if "rospy.init_node() has already been called" in str(e):
                        print(f"✅ ROS1 node already initialized: {self.node_name}")
                        self.node_initialized = True
                    else:
                        raise e
            
            # Initialize publishers
            if CUSTOM_MSGS_AVAILABLE:
                self.objects_publisher = rospy.Publisher(
                    self.objects_topic, DetectedObjects, queue_size=10
                )
                print(f"✅ Objects publisher: {self.objects_topic} (DetectedObjects)")
            else:
                self.objects_publisher = rospy.Publisher(
                    self.objects_topic, String, queue_size=10
                )
                print(f"⚠️  Objects publisher: {self.objects_topic} (String - fallback)")
            
            self.markers_publisher = rospy.Publisher(
                self.markers_topic, MarkerArray, queue_size=10
            )
            print(f"✅ Markers publisher: {self.markers_topic}")
            
            self.pointcloud_publisher = rospy.Publisher(
                self.pointcloud_topic, PointCloud2, queue_size=10
            )
            print(f"✅ PointCloud publisher: {self.pointcloud_topic}")
            
            self.camera_info_publisher = rospy.Publisher(
                self.camera_info_topic, CameraInfo, queue_size=10
            )
            print(f"✅ Camera info publisher: {self.camera_info_topic}")
            
            # Initialize TF
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()
            print(f"✅ TF2 initialized: {self.tf_frame} -> {self.world_frame}")
            
            # Wait for publishers to be ready
            rospy.sleep(0.5)
            
            self.connected = True
            print(f"🚀 ROS1 Noetic communication ready!")
            return True
            
        except Exception as e:
            print(f"❌ ROS1 initialization failed: {e}")
            return False
    
    def send_detection_data(self, detection_results: List[Dict]) -> bool:
        """
        Send detection data via ROS1 Noetic topics
        """
        if not self.connected:
            print("❌ Not connected to ROS1")
            return False
        
        try:
            current_time = rospy.Time.now()
            
            # Publish objects
            if self._publish_objects(detection_results, current_time):
                self.stats['objects_published'] += 1
            
            # Publish visualization markers
            if self._publish_markers(detection_results, current_time):
                self.stats['markers_published'] += 1
            
            # Publish point cloud
            if self._publish_pointcloud(detection_results, current_time):
                pass  # No separate counter for pointcloud
            
            # Publish camera TF
            if self._publish_camera_tf(current_time):
                self.stats['tf_published'] += 1
            
            self.stats['messages_sent'] += 1
            return True
            
        except Exception as e:
            print(f"❌ Failed to send detection data: {e}")
            return False
    
    def _publish_objects(self, detection_results: List[Dict], timestamp: rospy.Time) -> bool:
        """Publish detected objects"""
        try:
            if CUSTOM_MSGS_AVAILABLE:
                # Use custom message format
                objects_msg = DetectedObjects()
                objects_msg.header.stamp = timestamp
                objects_msg.header.frame_id = self.tf_frame
                objects_msg.frame_id = self.tf_frame
                objects_msg.object_count = len(detection_results)
                objects_msg.detection_enabled = True
                
                for result in detection_results:
                    obj_msg = DetectedObject()
                    obj_msg.object_id = result.get('tracker_id', 0)
                    obj_msg.class_name = result.get('class_name', 'unknown')
                    obj_msg.confidence = result.get('confidence', 0.0)
                    obj_msg.timestamp = timestamp
                    
                    # 2D bounding box
                    bbox = result.get('bbox', [0, 0, 0, 0])
                    obj_msg.bbox_2d = bbox
                    obj_msg.image_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    
                    # 3D position
                    pos_3d = self._calculate_3d_position(result)
                    obj_msg.position_3d.x = pos_3d[0]
                    obj_msg.position_3d.y = pos_3d[1]
                    obj_msg.position_3d.z = pos_3d[2]
                    obj_msg.depth_mm = result.get('depth_mm', 0.0)
                    
                    # Camera info
                    obj_msg.camera_frame = self.tf_frame
                    obj_msg.image_width = result.get('image_width', 640)
                    obj_msg.image_height = result.get('image_height', 480)
                    
                    # SLAM info
                    obj_msg.is_static = True
                    obj_msg.reliability_score = result.get('confidence', 0.0)
                    obj_msg.position_uncertainty = [0.1, 0.1, 0.1]  # Default uncertainty
                    
                    objects_msg.objects.append(obj_msg)
                
                # Calculate processing time (if available)
                objects_msg.processing_time_ms = result.get('processing_time_ms', 0.0)
                objects_msg.detection_fps = result.get('detection_fps', 0.0)
                
                self.objects_publisher.publish(objects_msg)
                
            else:
                # Fallback to String message
                data_packet = {
                    "header": {
                        "stamp": timestamp.to_sec(),
                        "frame_id": self.tf_frame
                    },
                    "object_count": len(detection_results),
                    "objects": []
                }
                
                for result in detection_results:
                    obj_data = {
                        "object_id": result.get('tracker_id', 0),
                        "class_name": result.get('class_name', 'unknown'),
                        "confidence": result.get('confidence', 0.0),
                        "bbox": result.get('bbox', [0, 0, 0, 0]),
                        "position_3d": self._calculate_3d_position(result),
                        "depth_mm": result.get('depth_mm', 0.0),
                        "is_static": True,
                        "reliability_score": result.get('confidence', 0.0)
                    }
                    data_packet["objects"].append(obj_data)
                
                msg = String()
                msg.data = json.dumps(data_packet)
                self.objects_publisher.publish(msg)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to publish objects: {e}")
            return False
    
    def _publish_markers(self, detection_results: List[Dict], timestamp: rospy.Time) -> bool:
        """Publish visualization markers for RViz"""
        try:
            marker_array = MarkerArray()
            
            for i, result in enumerate(detection_results):
                marker = Marker()
                marker.header.stamp = timestamp
                marker.header.frame_id = self.tf_frame
                marker.ns = "detected_objects"
                marker.id = result.get('tracker_id', i)
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                # Position
                pos_3d = self._calculate_3d_position(result)
                marker.pose.position.x = pos_3d[0]
                marker.pose.position.y = pos_3d[1]
                marker.pose.position.z = pos_3d[2]
                
                # Orientation (identity)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Scale based on object type
                scale = self._get_object_scale(result.get('class_name', 'unknown'))
                marker.scale.x = scale[0]
                marker.scale.y = scale[1]
                marker.scale.z = scale[2]
                
                # Color based on confidence
                confidence = result.get('confidence', 0.0)
                marker.color.r = 1.0 - confidence
                marker.color.g = confidence
                marker.color.b = 0.0
                marker.color.a = 0.7
                
                # Lifetime
                marker.lifetime = rospy.Duration(2.0)
                
                marker_array.markers.append(marker)
                
                # Add text marker
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "object_labels"
                text_marker.id = result.get('tracker_id', i) + 1000
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = pos_3d[0]
                text_marker.pose.position.y = pos_3d[1]
                text_marker.pose.position.z = pos_3d[2] + 0.3
                
                text_marker.scale.z = 0.2
                
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                text_marker.text = f"{result.get('class_name', 'unknown')}\n{confidence:.2f}"
                text_marker.lifetime = rospy.Duration(2.0)
                
                marker_array.markers.append(text_marker)
            
            self.markers_publisher.publish(marker_array)
            return True
            
        except Exception as e:
            print(f"❌ Failed to publish markers: {e}")
            return False
    
    def _publish_pointcloud(self, detection_results: List[Dict], timestamp: rospy.Time) -> bool:
        """Publish point cloud of detected objects"""
        try:
            # This is a simplified point cloud - you might want to use the actual depth data
            # For now, we'll create a point for each detected object
            
            points = []
            for result in detection_results:
                pos_3d = self._calculate_3d_position(result)
                points.append([pos_3d[0], pos_3d[1], pos_3d[2]])
            
            if not points:
                return True
            
            # Create PointCloud2 message
            header = Header()
            header.stamp = timestamp
            header.frame_id = self.tf_frame
            
            # Convert points to PointCloud2 format
            points_array = np.array(points, dtype=np.float32)
            
            # Create PointCloud2 message (simplified - you might want to add more fields)
            pointcloud_msg = PointCloud2()
            pointcloud_msg.header = header
            pointcloud_msg.height = 1
            pointcloud_msg.width = len(points)
            pointcloud_msg.fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]
            pointcloud_msg.is_bigendian = False
            pointcloud_msg.point_step = 12
            pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
            pointcloud_msg.data = points_array.tobytes()
            pointcloud_msg.is_dense = True
            
            self.pointcloud_publisher.publish(pointcloud_msg)
            return True
            
        except Exception as e:
            print(f"❌ Failed to publish pointcloud: {e}")
            return False
    
    def _publish_camera_tf(self, timestamp: rospy.Time) -> bool:
        """Publish camera transform"""
        try:
            # Publish static transform from camera to world frame
            # This is a simplified transform - you should use actual camera pose
            
            transform = TransformStamped()
            transform.header.stamp = timestamp
            transform.header.frame_id = self.world_frame
            transform.child_frame_id = self.tf_frame
            
            # Position (assuming camera is at origin for now)
            transform.transform.translation.x = 0.0
            transform.transform.translation.y = 0.0
            transform.transform.translation.z = 0.0
            
            # Orientation (identity)
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(transform)
            return True
            
        except Exception as e:
            print(f"❌ Failed to publish camera TF: {e}")
            return False
    
    def _calculate_3d_position(self, detection: Dict) -> List[float]:
        """Calculate 3D position from detection data"""
        depth_m = detection.get('depth_m', 0.0)
        if depth_m == 0.0:
            depth_m = detection.get('depth_mm', 0.0) / 1000.0
        
        bbox = detection.get('bbox', [0, 0, 0, 0])
        
        # Simplified 3D position calculation
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Convert image coordinates to 3D coordinates (simplified)
        fov_h = 60  # degrees
        fov_v = 45  # degrees
        img_w = detection.get('image_width', 640)
        img_h = detection.get('image_height', 480)
        
        # Calculate angles from center
        angle_h = (center_x - img_w/2) * fov_h / img_w
        angle_v = (center_y - img_h/2) * fov_v / img_h
        
        # Calculate 3D position
        x = depth_m * np.sin(np.radians(angle_h))
        y = depth_m * np.sin(np.radians(angle_v))
        z = depth_m * np.cos(np.radians(angle_h)) * np.cos(np.radians(angle_v))
        
        return [x, y, z]
    
    def _get_object_scale(self, class_name: str) -> List[float]:
        """Get object scale for visualization"""
        scale_map = {
            'person': [0.6, 0.6, 1.8],
            'chair': [0.6, 0.6, 1.0],
            'table': [1.2, 0.8, 0.8],
            'bottle': [0.1, 0.1, 0.3],
            'cup': [0.1, 0.1, 0.15],
            'laptop': [0.4, 0.3, 0.03],
            'book': [0.2, 0.3, 0.03],
            'plant': [0.3, 0.3, 0.6],
            'monitor': [0.5, 0.4, 0.4],
            'default': [0.2, 0.2, 0.2]
        }
        
        return scale_map.get(class_name.lower(), scale_map['default'])
    
    def get_statistics(self) -> Dict:
        """Get communication statistics"""
        runtime = time.time() - self.stats['start_time']
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'messages_per_second': self.stats['messages_sent'] / runtime if runtime > 0 else 0,
            'objects_per_second': self.stats['objects_published'] / runtime if runtime > 0 else 0
        }
    
    def disconnect(self):
        """Shutdown ROS communication"""
        if self.connected:
            self.connected = False
            print("🔌 ROS1 Noetic communication disconnected")
    
    def is_connected(self) -> bool:
        """Check if ROS is connected and node is running"""
        return self.connected and not rospy.is_shutdown()

def create_ros_noetic_handler(**kwargs) -> ROS1NoeticCommunication:
    """Factory function to create ROS1 Noetic communication handler"""
    handler = ROS1NoeticCommunication(**kwargs)
    
    if handler.connect():
        return handler
    else:
        raise RuntimeError("Failed to initialize ROS1 Noetic communication")

if __name__ == "__main__":
    # Example usage
    print("🧪 Testing ROS1 Noetic Communication Handler")
    
    try:
        handler = create_ros_noetic_handler()
        
        # Simulate detection data
        test_data = [{
            'tracker_id': 1,
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': [100, 150, 200, 300],
            'depth_mm': 2500,
            'depth_m': 2.5,
            'image_width': 640,
            'image_height': 480
        }]
        
        # Send test data
        print("📡 Sending test detection data...")
        for i in range(10):
            if handler.send_detection_data(test_data):
                print(f"✅ Test message {i+1} sent successfully")
            else:
                print(f"❌ Failed to send test message {i+1}")
            
            time.sleep(1.0)
        
        # Print statistics
        print("📊 Final Statistics:", handler.get_statistics())
        
        handler.disconnect()
        
    except Exception as e:
        print(f"❌ Test failed: {e}") 