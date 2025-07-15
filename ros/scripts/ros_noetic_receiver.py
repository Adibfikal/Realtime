#!/usr/bin/env python3
"""
ROS1 Noetic Navigation System Receiver
Receives object detection data from the realtime detection system via ROS1 Noetic
This is what your teammate's navigation system would use
"""

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading

# ROS1 Noetic message imports
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage

# Try to import custom messages
try:
    from realtime_detection_msgs.msg import DetectedObject, DetectedObjects
    CUSTOM_MSGS_AVAILABLE = True
    print("✅ Custom detection messages available")
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False
    print("⚠️  Custom detection messages not available, using String messages")

class ROS1NoeticReceiver:
    """
    ROS1 Noetic receiver for navigation system integration
    Receives object detection data and processes it for SLAM/Navigation
    """
    
    def __init__(self, 
                 node_name: str = "navigation_receiver",
                 objects_topic: str = "/detected_objects",
                 markers_topic: str = "/detection_markers",
                 pointcloud_topic: str = "/detection_pointcloud",
                 landmarks_topic: str = "/navigation_landmarks",
                 obstacles_topic: str = "/navigation_obstacles",
                 tf_frame: str = "camera_link",
                 world_frame: str = "map"):
        
        self.node_name = node_name
        self.objects_topic = objects_topic
        self.markers_topic = markers_topic
        self.pointcloud_topic = pointcloud_topic
        self.landmarks_topic = landmarks_topic
        self.obstacles_topic = obstacles_topic
        self.tf_frame = tf_frame
        self.world_frame = world_frame
        
        # Data storage
        self.received_objects = deque(maxlen=1000)
        self.landmarks = {}
        self.obstacles = {}
        self.object_history = {}
        
        # ROS components
        self.tf_buffer = None
        self.tf_listener = None
        self.subscribers = {}
        self.publishers = {}
        
        # Processing parameters
        self.landmark_distance_threshold = 1.0  # meters
        self.min_observations = 3
        self.landmark_timeout = 30.0  # seconds
        self.processing_enabled = True
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'objects_processed': 0,
            'landmarks_created': 0,
            'obstacles_detected': 0,
            'start_time': time.time()
        }
        
        # Processing thread
        self.processing_thread = None
        self.running = False
        
        print(f"🚀 ROS1 Noetic Navigation Receiver initialized")
        print(f"📡 Node: {node_name}")
        print(f"🎯 Listening on: {objects_topic}")
        print(f"📍 Publishing landmarks to: {landmarks_topic}")
        print(f"🚧 Publishing obstacles to: {obstacles_topic}")
    
    def start(self) -> bool:
        """Start the ROS1 Noetic receiver"""
        try:
            # Initialize ROS node
            rospy.init_node(self.node_name, anonymous=True)
            print(f"✅ ROS1 node started: {self.node_name}")
            
            # Initialize TF
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            print("✅ TF2 listener initialized")
            
            # Setup subscribers
            self._setup_subscribers()
            
            # Setup publishers
            self._setup_publishers()
            
            # Start processing thread
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.processing_thread.start()
            print("✅ Processing thread started")
            
            print("🔄 Navigation receiver ready!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start receiver: {e}")
            return False
    
    def _setup_subscribers(self):
        """Setup ROS subscribers"""
        try:
            if CUSTOM_MSGS_AVAILABLE:
                self.subscribers['objects'] = rospy.Subscriber(
                    self.objects_topic, DetectedObjects, self._objects_callback
                )
                print(f"✅ Subscribed to {self.objects_topic} (DetectedObjects)")
            else:
                self.subscribers['objects'] = rospy.Subscriber(
                    self.objects_topic, String, self._objects_string_callback
                )
                print(f"⚠️  Subscribed to {self.objects_topic} (String - fallback)")
            
            # Subscribe to markers and pointcloud for additional data
            self.subscribers['markers'] = rospy.Subscriber(
                self.markers_topic, MarkerArray, self._markers_callback
            )
            print(f"✅ Subscribed to {self.markers_topic}")
            
            self.subscribers['pointcloud'] = rospy.Subscriber(
                self.pointcloud_topic, PointCloud2, self._pointcloud_callback
            )
            print(f"✅ Subscribed to {self.pointcloud_topic}")
            
        except Exception as e:
            print(f"❌ Failed to setup subscribers: {e}")
    
    def _setup_publishers(self):
        """Setup ROS publishers for navigation data"""
        try:
            # Publish processed landmarks for navigation
            self.publishers['landmarks'] = rospy.Publisher(
                self.landmarks_topic, MarkerArray, queue_size=10
            )
            print(f"✅ Publishing landmarks to {self.landmarks_topic}")
            
            # Publish obstacles for path planning
            self.publishers['obstacles'] = rospy.Publisher(
                self.obstacles_topic, MarkerArray, queue_size=10
            )
            print(f"✅ Publishing obstacles to {self.obstacles_topic}")
            
            # Publish occupancy grid (optional)
            self.publishers['occupancy'] = rospy.Publisher(
                "/navigation_occupancy_grid", OccupancyGrid, queue_size=1
            )
            print(f"✅ Publishing occupancy grid to /navigation_occupancy_grid")
            
        except Exception as e:
            print(f"❌ Failed to setup publishers: {e}")
    
    def _objects_callback(self, msg: DetectedObjects):
        """Callback for custom DetectedObjects messages"""
        try:
            current_time = time.time()
            
            # Process each detected object
            for obj in msg.objects:
                # Convert to standard format
                obj_data = {
                    'object_id': obj.object_id,
                    'class_name': obj.class_name,
                    'confidence': obj.confidence,
                    'timestamp': obj.timestamp.to_sec(),
                    'bbox_2d': list(obj.bbox_2d),
                    'image_center': list(obj.image_center),
                    'position_3d': [obj.position_3d.x, obj.position_3d.y, obj.position_3d.z],
                    'depth_mm': obj.depth_mm,
                    'camera_frame': obj.camera_frame,
                    'is_static': obj.is_static,
                    'reliability_score': obj.reliability_score,
                    'position_uncertainty': list(obj.position_uncertainty),
                    'received_time': current_time
                }
                
                # Transform to world coordinates if possible
                world_position = self._transform_to_world_frame(obj_data)
                if world_position:
                    obj_data['world_position'] = world_position
                
                self.received_objects.append(obj_data)
            
            self.stats['messages_received'] += 1
            self.stats['objects_processed'] += len(msg.objects)
            
            # Print periodic statistics
            if self.stats['messages_received'] % 10 == 0:
                self._print_statistics()
                
        except Exception as e:
            print(f"❌ Error processing objects message: {e}")
    
    def _objects_string_callback(self, msg: String):
        """Callback for String messages (fallback)"""
        try:
            data = json.loads(msg.data)
            current_time = time.time()
            
            # Process each detected object
            for obj in data.get('objects', []):
                obj_data = {
                    'object_id': obj.get('object_id', 0),
                    'class_name': obj.get('class_name', 'unknown'),
                    'confidence': obj.get('confidence', 0.0),
                    'timestamp': current_time,
                    'bbox_2d': obj.get('bbox', [0, 0, 0, 0]),
                    'position_3d': obj.get('position_3d', [0, 0, 0]),
                    'depth_mm': obj.get('depth_mm', 0.0),
                    'camera_frame': self.tf_frame,
                    'is_static': obj.get('is_static', True),
                    'reliability_score': obj.get('reliability_score', 0.0),
                    'received_time': current_time
                }
                
                # Transform to world coordinates if possible
                world_position = self._transform_to_world_frame(obj_data)
                if world_position:
                    obj_data['world_position'] = world_position
                
                self.received_objects.append(obj_data)
            
            self.stats['messages_received'] += 1
            self.stats['objects_processed'] += len(data.get('objects', []))
            
            # Print periodic statistics
            if self.stats['messages_received'] % 10 == 0:
                self._print_statistics()
                
        except Exception as e:
            print(f"❌ Error processing string message: {e}")
    
    def _markers_callback(self, msg: MarkerArray):
        """Callback for visualization markers"""
        # This is mainly for visualization, but we could use it for additional processing
        pass
    
    def _pointcloud_callback(self, msg: PointCloud2):
        """Callback for point cloud data"""
        # This could be used for additional spatial processing
        pass
    
    def _transform_to_world_frame(self, obj_data: Dict) -> Optional[List[float]]:
        """Transform object position from camera frame to world frame"""
        try:
            if not self.tf_buffer:
                return None
            
            # Create point in camera frame
            point_camera = PoseStamped()
            point_camera.header.frame_id = self.tf_frame
            point_camera.header.stamp = rospy.Time(0)  # Use latest available transform
            
            pos_3d = obj_data['position_3d']
            point_camera.pose.position.x = pos_3d[0]
            point_camera.pose.position.y = pos_3d[1]
            point_camera.pose.position.z = pos_3d[2]
            point_camera.pose.orientation.w = 1.0
            
            # Transform to world frame
            point_world = self.tf_buffer.transform(point_camera, self.world_frame, rospy.Duration(1.0))
            
            return [
                point_world.pose.position.x,
                point_world.pose.position.y,
                point_world.pose.position.z
            ]
            
        except Exception as e:
            # Transform failed, return None
            return None
    
    def _processing_worker(self):
        """Background processing worker"""
        while self.running and not rospy.is_shutdown():
            try:
                # Process recent objects for landmark creation
                self._process_landmarks()
                
                # Process obstacles for navigation
                self._process_obstacles()
                
                # Publish navigation data
                self._publish_navigation_data()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep
                time.sleep(1.0)
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
    
    def _process_landmarks(self):
        """Process objects to create landmarks for SLAM"""
        try:
            # Group recent objects by ID
            current_time = time.time()
            recent_objects = {}
            
            for obj in self.received_objects:
                if current_time - obj['received_time'] < 5.0:  # Last 5 seconds
                    obj_id = obj['object_id']
                    if obj_id not in recent_objects:
                        recent_objects[obj_id] = []
                    recent_objects[obj_id].append(obj)
            
            # Create/update landmarks
            for obj_id, objects in recent_objects.items():
                if len(objects) >= self.min_observations:
                    self._create_or_update_landmark(obj_id, objects)
                    
        except Exception as e:
            print(f"❌ Landmark processing error: {e}")
    
    def _create_or_update_landmark(self, obj_id: int, objects: List[Dict]):
        """Create or update a landmark"""
        try:
            # Use world position if available, otherwise camera position
            positions = []
            confidences = []
            
            for obj in objects:
                if 'world_position' in obj:
                    positions.append(obj['world_position'])
                else:
                    positions.append(obj['position_3d'])
                confidences.append(obj['confidence'])
            
            # Calculate average position and confidence
            avg_position = np.mean(positions, axis=0).tolist()
            avg_confidence = np.mean(confidences)
            
            # Create landmark
            landmark = {
                'id': obj_id,
                'type': objects[0]['class_name'],
                'position': avg_position,
                'confidence': avg_confidence,
                'observations': len(objects),
                'last_seen': time.time(),
                'uncertainty': np.std(positions, axis=0).tolist() if len(positions) > 1 else [0.1, 0.1, 0.1],
                'is_static': objects[0]['is_static']
            }
            
            # Add to landmarks
            if obj_id not in self.landmarks:
                self.stats['landmarks_created'] += 1
                print(f"🏷️  New landmark: {landmark['type']} at {avg_position}")
            
            self.landmarks[obj_id] = landmark
            
        except Exception as e:
            print(f"❌ Landmark creation error: {e}")
    
    def _process_obstacles(self):
        """Process objects to create obstacles for path planning"""
        try:
            current_time = time.time()
            
            # Clear old obstacles
            self.obstacles = {}
            
            # Create obstacles from high-confidence landmarks
            for landmark_id, landmark in self.landmarks.items():
                if (landmark['confidence'] > 0.7 and 
                    current_time - landmark['last_seen'] < 10.0):
                    
                    obstacle = {
                        'id': landmark_id,
                        'type': landmark['type'],
                        'position': landmark['position'],
                        'radius': self._get_obstacle_radius(landmark['type']),
                        'confidence': landmark['confidence'],
                        'last_seen': landmark['last_seen']
                    }
                    
                    self.obstacles[landmark_id] = obstacle
                    self.stats['obstacles_detected'] = len(self.obstacles)
                    
        except Exception as e:
            print(f"❌ Obstacle processing error: {e}")
    
    def _get_obstacle_radius(self, object_type: str) -> float:
        """Get obstacle radius for path planning"""
        radius_map = {
            'person': 0.5,
            'chair': 0.6,
            'table': 0.8,
            'bottle': 0.1,
            'cup': 0.1,
            'laptop': 0.3,
            'book': 0.2,
            'plant': 0.3,
            'monitor': 0.4,
            'default': 0.3
        }
        
        return radius_map.get(object_type.lower(), radius_map['default'])
    
    def _publish_navigation_data(self):
        """Publish processed navigation data"""
        try:
            current_time = rospy.Time.now()
            
            # Publish landmarks
            self._publish_landmarks_markers(current_time)
            
            # Publish obstacles
            self._publish_obstacles_markers(current_time)
            
            # Publish occupancy grid (optional)
            self._publish_occupancy_grid(current_time)
            
        except Exception as e:
            print(f"❌ Navigation data publishing error: {e}")
    
    def _publish_landmarks_markers(self, timestamp: rospy.Time):
        """Publish landmarks as markers"""
        try:
            marker_array = MarkerArray()
            
            for landmark in self.landmarks.values():
                marker = Marker()
                marker.header.stamp = timestamp
                marker.header.frame_id = self.world_frame
                marker.ns = "navigation_landmarks"
                marker.id = landmark['id']
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                # Position
                marker.pose.position.x = landmark['position'][0]
                marker.pose.position.y = landmark['position'][1]
                marker.pose.position.z = landmark['position'][2]
                
                # Orientation
                marker.pose.orientation.w = 1.0
                
                # Scale
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.5
                
                # Color (blue for landmarks)
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 0.8
                
                marker.lifetime = rospy.Duration(5.0)
                
                marker_array.markers.append(marker)
            
            self.publishers['landmarks'].publish(marker_array)
            
        except Exception as e:
            print(f"❌ Landmark markers publishing error: {e}")
    
    def _publish_obstacles_markers(self, timestamp: rospy.Time):
        """Publish obstacles as markers"""
        try:
            marker_array = MarkerArray()
            
            for obstacle in self.obstacles.values():
                marker = Marker()
                marker.header.stamp = timestamp
                marker.header.frame_id = self.world_frame
                marker.ns = "navigation_obstacles"
                marker.id = obstacle['id'] + 1000
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                # Position
                marker.pose.position.x = obstacle['position'][0]
                marker.pose.position.y = obstacle['position'][1]
                marker.pose.position.z = obstacle['position'][2]
                
                # Orientation
                marker.pose.orientation.w = 1.0
                
                # Scale based on obstacle radius
                radius = obstacle['radius']
                marker.scale.x = radius * 2
                marker.scale.y = radius * 2
                marker.scale.z = 0.1
                
                # Color (red for obstacles)
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.6
                
                marker.lifetime = rospy.Duration(5.0)
                
                marker_array.markers.append(marker)
            
            self.publishers['obstacles'].publish(marker_array)
            
        except Exception as e:
            print(f"❌ Obstacle markers publishing error: {e}")
    
    def _publish_occupancy_grid(self, timestamp: rospy.Time):
        """Publish occupancy grid for path planning"""
        try:
            # Create a simple occupancy grid based on obstacles
            # This is a simplified version - you might want a more sophisticated approach
            
            grid_size = 100  # 100x100 grid
            resolution = 0.1  # 10cm per cell
            
            # Create occupancy grid
            occupancy_grid = OccupancyGrid()
            occupancy_grid.header.stamp = timestamp
            occupancy_grid.header.frame_id = self.world_frame
            
            # Map metadata
            occupancy_grid.info.resolution = resolution
            occupancy_grid.info.width = grid_size
            occupancy_grid.info.height = grid_size
            occupancy_grid.info.origin.position.x = -grid_size * resolution / 2
            occupancy_grid.info.origin.position.y = -grid_size * resolution / 2
            occupancy_grid.info.origin.position.z = 0
            occupancy_grid.info.origin.orientation.w = 1.0
            
            # Initialize grid (0 = free, 100 = occupied, -1 = unknown)
            grid_data = [0] * (grid_size * grid_size)
            
            # Add obstacles to grid
            for obstacle in self.obstacles.values():
                x = obstacle['position'][0]
                y = obstacle['position'][1]
                radius = obstacle['radius']
                
                # Convert to grid coordinates
                grid_x = int((x - occupancy_grid.info.origin.position.x) / resolution)
                grid_y = int((y - occupancy_grid.info.origin.position.y) / resolution)
                grid_radius = int(radius / resolution)
                
                # Mark cells as occupied
                for i in range(max(0, grid_x - grid_radius), min(grid_size, grid_x + grid_radius + 1)):
                    for j in range(max(0, grid_y - grid_radius), min(grid_size, grid_y + grid_radius + 1)):
                        if i < grid_size and j < grid_size:
                            distance = np.sqrt((i - grid_x)**2 + (j - grid_y)**2)
                            if distance <= grid_radius:
                                grid_data[j * grid_size + i] = 100  # Occupied
            
            occupancy_grid.data = grid_data
            self.publishers['occupancy'].publish(occupancy_grid)
            
        except Exception as e:
            print(f"❌ Occupancy grid publishing error: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old landmarks and data"""
        try:
            current_time = time.time()
            
            # Remove old landmarks
            to_remove = []
            for landmark_id, landmark in self.landmarks.items():
                if current_time - landmark['last_seen'] > self.landmark_timeout:
                    to_remove.append(landmark_id)
            
            for landmark_id in to_remove:
                del self.landmarks[landmark_id]
                
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
    
    def _print_statistics(self):
        """Print receiver statistics"""
        runtime = time.time() - self.stats['start_time']
        print(f"\n📊 Navigation Receiver Statistics:")
        print(f"   Messages received: {self.stats['messages_received']}")
        print(f"   Objects processed: {self.stats['objects_processed']}")
        print(f"   Landmarks created: {self.stats['landmarks_created']}")
        print(f"   Active landmarks: {len(self.landmarks)}")
        print(f"   Active obstacles: {len(self.obstacles)}")
        print(f"   Runtime: {runtime:.1f}s")
        print(f"   Messages/sec: {self.stats['messages_received'] / runtime:.2f}")
    
    def get_landmarks_for_slam(self) -> List[Dict]:
        """Get landmarks formatted for SLAM system"""
        slam_landmarks = []
        
        for landmark in self.landmarks.values():
            if landmark['observations'] >= self.min_observations:
                slam_landmark = {
                    'id': landmark['id'],
                    'type': landmark['type'],
                    'position': landmark['position'],
                    'confidence': landmark['confidence'],
                    'uncertainty': np.linalg.norm(landmark['uncertainty']),
                    'observations': landmark['observations'],
                    'is_static': landmark['is_static']
                }
                slam_landmarks.append(slam_landmark)
        
        return slam_landmarks
    
    def get_obstacles_for_planning(self) -> List[Dict]:
        """Get obstacles formatted for path planning"""
        planning_obstacles = []
        
        for obstacle in self.obstacles.values():
            planning_obstacle = {
                'id': obstacle['id'],
                'type': obstacle['type'],
                'position': obstacle['position'],
                'radius': obstacle['radius'],
                'confidence': obstacle['confidence']
            }
            planning_obstacles.append(planning_obstacle)
        
        return planning_obstacles
    
    def stop(self):
        """Stop the receiver"""
        print("\n🛑 Stopping navigation receiver...")
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        print("✅ Navigation receiver stopped")

def main():
    """Main function for running the receiver"""
    try:
        print("🚀 Starting ROS1 Noetic Navigation Receiver...")
        
        receiver = ROS1NoeticReceiver()
        
        if receiver.start():
            print("✅ Receiver started successfully")
            print("📡 Listening for object detection data...")
            print("Press Ctrl+C to stop")
            
            # Keep running
            rospy.spin()
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'receiver' in locals():
            receiver.stop()

if __name__ == "__main__":
    main() 