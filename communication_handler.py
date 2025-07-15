"""
Communication Handler for SLAM Integration
Handles sending object detection data to navigation system via multiple protocols
"""

import json
import socket
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

# Try to import ROS if available
try:
    import rospy
    from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, String
    import tf2_ros
    import tf2_geometry_msgs
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Try to import ROS1 Noetic specialized communication
try:
    from ros.scripts.ros_noetic_communication import ROS1NoeticCommunication
    ROS_NOETIC_AVAILABLE = True
except ImportError:
    ROS_NOETIC_AVAILABLE = False

class CommunicationProtocol(Enum):
    """Supported communication protocols"""
    SOCKET = "socket"
    ROS = "ros"
    ROS_NOETIC = "ros_noetic"
    MESSAGE_QUEUE = "message_queue"
    FILE = "file"

@dataclass
class DetectedObject:
    """Standardized object detection data structure for SLAM"""
    # Object identification
    object_id: int
    class_name: str
    confidence: float
    timestamp: float
    
    # 2D information (image coordinates)
    bbox_2d: List[float]  # [x1, y1, x2, y2]
    image_center: List[float]  # [x, y] center of bbox in image
    
    # 3D information (camera coordinates)
    position_3d: List[float]  # [x, y, z] in meters from camera
    depth_mm: float
    
    # Camera parameters
    camera_frame: str = "camera_link"
    image_width: int = 640
    image_height: int = 480
    
    # Additional SLAM-relevant data
    is_static: bool = True  # Assume objects are static landmarks
    reliability_score: float = 1.0  # How reliable this detection is
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_world_coordinates(self, camera_pose: Optional[List[float]] = None) -> List[float]:
        """
        Convert object position from camera coordinates to world coordinates
        camera_pose: [x, y, z, qx, qy, qz, qw] (position + quaternion)
        """
        if camera_pose is None:
            return self.position_3d
        
        # Simple transformation (would need proper rotation matrix in real implementation)
        # This is a placeholder - you'd need proper transformation based on camera pose
        world_x = camera_pose[0] + self.position_3d[0]
        world_y = camera_pose[1] + self.position_3d[1]
        world_z = camera_pose[2] + self.position_3d[2]
        
        return [world_x, world_y, world_z]

class CommunicationInterface(ABC):
    """Abstract base class for communication protocols"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection"""
        pass
    
    @abstractmethod
    def send_detection_data(self, objects: List[DetectedObject]) -> bool:
        """Send detection data"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active"""
        pass

class SocketCommunication(CommunicationInterface):
    """Socket-based communication for real-time data transfer"""
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to socket server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✓ Connected to socket server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Socket connection failed: {e}")
            self.connected = False
            return False
    
    def send_detection_data(self, objects: List[DetectedObject]) -> bool:
        """Send detection data via socket"""
        if not self.connected or not self.socket:
            return False
        
        try:
            # Prepare data packet
            data_packet = {
                "timestamp": time.time(),
                "frame_id": "camera_link",
                "object_count": len(objects),
                "objects": [obj.to_dict() for obj in objects]
            }
            
            # Serialize and send
            message = json.dumps(data_packet) + "\n"
            self.socket.send(message.encode('utf-8'))
            return True
            
        except Exception as e:
            print(f"✗ Socket send failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close socket connection"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("Socket connection closed")
    
    def is_connected(self) -> bool:
        return self.connected

class ROSCommunication(CommunicationInterface):
    """ROS-based communication for robotics integration"""
    
    def __init__(self, topic_name: str = "/detected_objects", node_name: str = "object_detector"):
        if not ROS_AVAILABLE:
            raise ImportError("ROS not available. Install ROS packages to use this communication method.")
        
        self.topic_name = topic_name
        self.node_name = node_name
        self.publisher = None
        self.tf_buffer = None
        self.tf_listener = None
        self.connected = False
        
    def connect(self) -> bool:
        """Initialize ROS node and publisher"""
        try:
            if not rospy.get_node_uri():
                rospy.init_node(self.node_name, anonymous=True)
            
            # Create publisher for detection data
            self.publisher = rospy.Publisher(self.topic_name, String, queue_size=10)
            
            # Initialize TF for coordinate transformations
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            
            self.connected = True
            print(f"✓ ROS node initialized: {self.node_name}")
            print(f"✓ Publishing to topic: {self.topic_name}")
            return True
            
        except Exception as e:
            print(f"✗ ROS initialization failed: {e}")
            return False
    
    def send_detection_data(self, objects: List[DetectedObject]) -> bool:
        """Send detection data via ROS topic"""
        if not self.connected or not self.publisher:
            return False
        
        try:
            # Transform objects to world coordinates if possible
            transformed_objects = []
            for obj in objects:
                try:
                    # Try to get transform from camera to world frame
                    if self.tf_buffer:
                        transform = self.tf_buffer.lookup_transform(
                            'map', obj.camera_frame, rospy.Time(0), rospy.Duration(1.0)
                        )
                        # Apply transformation (simplified)
                        world_pos = obj.to_world_coordinates()
                        obj_dict = obj.to_dict()
                        obj_dict['world_position'] = world_pos
                        transformed_objects.append(obj_dict)
                    else:
                        transformed_objects.append(obj.to_dict())
                except:
                    # If transform fails, use camera coordinates
                    transformed_objects.append(obj.to_dict())
            
            # Create ROS message
            message = String()
            message.data = json.dumps({
                "header": {
                    "stamp": rospy.Time.now().to_sec(),
                    "frame_id": "camera_link"
                },
                "objects": transformed_objects
            })
            
            # Publish message
            self.publisher.publish(message)
            return True
            
        except Exception as e:
            print(f"✗ ROS publish failed: {e}")
            return False
    
    def disconnect(self):
        """Shutdown ROS node"""
        if self.connected:
            self.connected = False
            print("ROS communication closed")
    
    def is_connected(self) -> bool:
        return self.connected and not rospy.is_shutdown()

class FileCommunication(CommunicationInterface):
    """File-based communication for offline processing"""
    
    def __init__(self, output_file: str = "detection_data.json"):
        self.output_file = output_file
        self.connected = False
        self.detection_buffer = []
        
    def connect(self) -> bool:
        """Initialize file communication"""
        try:
            # Create/clear the output file
            with open(self.output_file, 'w') as f:
                f.write("")
            self.connected = True
            print(f"✓ File communication initialized: {self.output_file}")
            return True
        except Exception as e:
            print(f"✗ File initialization failed: {e}")
            return False
    
    def send_detection_data(self, objects: List[DetectedObject]) -> bool:
        """Append detection data to file"""
        if not self.connected:
            return False
        
        try:
            data_packet = {
                "timestamp": time.time(),
                "objects": [obj.to_dict() for obj in objects]
            }
            
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(data_packet) + "\n")
            
            return True
            
        except Exception as e:
            print(f"✗ File write failed: {e}")
            return False
    
    def disconnect(self):
        """Close file communication"""
        self.connected = False
        print(f"File communication closed: {self.output_file}")
    
    def is_connected(self) -> bool:
        return self.connected

class CommunicationHandler:
    """Main communication handler with multiple protocol support"""
    
    def __init__(self, protocol: CommunicationProtocol = CommunicationProtocol.SOCKET):
        self.protocol = protocol
        self.communication = None
        self.is_running = False
        self.send_queue = queue.Queue()
        self.sender_thread = None
        self.stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'total_objects_sent': 0,
            'start_time': time.time()
        }
        
    def initialize(self, **kwargs) -> bool:
        """Initialize communication based on selected protocol"""
        try:
            if self.protocol == CommunicationProtocol.SOCKET:
                host = kwargs.get('host', 'localhost')
                port = kwargs.get('port', 8888)
                self.communication = SocketCommunication(host, port)
                
            elif self.protocol == CommunicationProtocol.ROS:
                topic = kwargs.get('topic', '/detected_objects')
                node = kwargs.get('node_name', 'object_detector')
                self.communication = ROSCommunication(topic, node)
                
            elif self.protocol == CommunicationProtocol.ROS_NOETIC:
                if not ROS_NOETIC_AVAILABLE:
                    raise ImportError("ROS1 Noetic communication not available. Install ROS1 Noetic and build custom messages.")
                
                # ROS1 Noetic specific parameters
                node_name = kwargs.get('node_name', 'realtime_object_detector')
                objects_topic = kwargs.get('objects_topic', '/detected_objects')
                markers_topic = kwargs.get('markers_topic', '/detection_markers')
                pointcloud_topic = kwargs.get('pointcloud_topic', '/detection_pointcloud')
                tf_frame = kwargs.get('tf_frame', 'camera_link')
                world_frame = kwargs.get('world_frame', 'map')
                
                self.communication = ROS1NoeticCommunication(
                    node_name=node_name,
                    objects_topic=objects_topic,
                    markers_topic=markers_topic,
                    pointcloud_topic=pointcloud_topic,
                    tf_frame=tf_frame,
                    world_frame=world_frame
                )
                
            elif self.protocol == CommunicationProtocol.FILE:
                output_file = kwargs.get('output_file', 'detection_data.json')
                self.communication = FileCommunication(output_file)
                
            else:
                print(f"Unsupported protocol: {self.protocol}")
                return False
            
            if self.communication and self.communication.connect():
                self.is_running = True
                self._start_sender_thread()
                print(f"✓ Communication handler initialized with {self.protocol.value}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"✗ Communication initialization failed: {e}")
            return False
    
    def _start_sender_thread(self):
        """Start background thread for sending data"""
        self.sender_thread = threading.Thread(target=self._sender_worker, daemon=True)
        self.sender_thread.start()
    
    def _sender_worker(self):
        """Background worker for sending queued data"""
        while self.is_running:
            try:
                # Get data from queue with timeout
                objects = self.send_queue.get(timeout=1.0)
                
                if objects and self.communication:
                    if self.communication.send_detection_data(objects):
                        self.stats['messages_sent'] += 1
                        self.stats['total_objects_sent'] += len(objects)
                    else:
                        self.stats['messages_failed'] += 1
                        # Try to reconnect if send failed
                        if not self.communication.is_connected():
                            print("Connection lost, attempting to reconnect...")
                            self.communication.connect()
                
                self.send_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Sender worker error: {e}")
                
    def send_detection_data(self, detection_results: List[Dict]) -> bool:
        """
        Send detection data to navigation system
        detection_results: List of detection dictionaries from object_detection_processor
        """
        if not self.is_running or not self.communication:
            return False
        
        try:
            # Convert detection results to DetectedObject format
            detected_objects = []
            for result in detection_results:
                obj = DetectedObject(
                    object_id=result.get('tracker_id', 0),
                    class_name=result.get('class_name', 'unknown'),
                    confidence=result.get('confidence', 0.0),
                    timestamp=time.time(),
                    bbox_2d=result.get('bbox', [0, 0, 0, 0]),
                    image_center=self._calculate_bbox_center(result.get('bbox', [0, 0, 0, 0])),
                    position_3d=self._calculate_3d_position(result),
                    depth_mm=result.get('depth_mm', 0.0)
                )
                detected_objects.append(obj)
            
            # Add to send queue (non-blocking)
            self.send_queue.put(detected_objects, block=False)
            return True
            
        except Exception as e:
            print(f"Failed to queue detection data: {e}")
            return False
    
    def _calculate_bbox_center(self, bbox: List[float]) -> List[float]:
        """Calculate center point of bounding box"""
        if len(bbox) < 4:
            return [0.0, 0.0]
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    def _calculate_3d_position(self, detection: Dict) -> List[float]:
        """
        Calculate 3D position from detection data
        This is a simplified version - you may need camera calibration for accurate results
        """
        depth_m = detection.get('depth_m', 0.0)
        bbox = detection.get('bbox', [0, 0, 0, 0])
        
        # Simplified 3D position calculation (assumes camera at origin)
        # In a real implementation, you'd use camera intrinsics and extrinsics
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Convert image coordinates to 3D coordinates (simplified)
        # Assuming camera FOV and image dimensions
        fov_h = 60  # degrees
        fov_v = 45  # degrees
        img_w = 640
        img_h = 480
        
        # Calculate angles from center
        angle_h = (center_x - img_w/2) * fov_h / img_w
        angle_v = (center_y - img_h/2) * fov_v / img_h
        
        # Calculate 3D position
        x = depth_m * np.sin(np.radians(angle_h))
        y = depth_m * np.sin(np.radians(angle_v))
        z = depth_m * np.cos(np.radians(angle_h)) * np.cos(np.radians(angle_v))
        
        return [x, y, z]
    
    def get_statistics(self) -> Dict:
        """Get communication statistics"""
        runtime = time.time() - self.stats['start_time']
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'messages_per_second': self.stats['messages_sent'] / runtime if runtime > 0 else 0,
            'objects_per_second': self.stats['total_objects_sent'] / runtime if runtime > 0 else 0,
            'success_rate': self.stats['messages_sent'] / (self.stats['messages_sent'] + self.stats['messages_failed']) if (self.stats['messages_sent'] + self.stats['messages_failed']) > 0 else 0
        }
    
    def shutdown(self):
        """Shutdown communication handler"""
        self.is_running = False
        
        if self.sender_thread:
            self.sender_thread.join(timeout=2.0)
        
        if self.communication:
            self.communication.disconnect()
        
        print("Communication handler shutdown complete")

# Example usage and configuration
def create_communication_handler(protocol: str = "socket", **kwargs) -> CommunicationHandler:
    """Factory function to create communication handler"""
    protocol_map = {
        "socket": CommunicationProtocol.SOCKET,
        "ros": CommunicationProtocol.ROS,
        "ros_noetic": CommunicationProtocol.ROS_NOETIC,
        "file": CommunicationProtocol.FILE
    }
    
    if protocol not in protocol_map:
        raise ValueError(f"Unsupported protocol: {protocol}")
    
    handler = CommunicationHandler(protocol_map[protocol])
    
    if handler.initialize(**kwargs):
        return handler
    else:
        raise RuntimeError(f"Failed to initialize {protocol} communication")

if __name__ == "__main__":
    # Example usage
    print("Communication Handler Test")
    
    # Test socket communication
    try:
        handler = create_communication_handler("socket", host="localhost", port=8888)
        
        # Simulate detection data
        test_data = [{
            'tracker_id': 1,
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': [100, 150, 200, 300],
            'depth_mm': 2500,
            'depth_m': 2.5
        }]
        
        # Send test data
        if handler.send_detection_data(test_data):
            print("✓ Test data sent successfully")
        else:
            print("✗ Failed to send test data")
        
        # Print statistics
        print("Statistics:", handler.get_statistics())
        
        # Keep running for a few seconds
        time.sleep(3)
        
        handler.shutdown()
        
    except Exception as e:
        print(f"Test failed: {e}") 