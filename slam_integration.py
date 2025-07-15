"""
SLAM Integration Module for Object Detection Data
Processes object detection data for navigation and SLAM systems
"""

import numpy as np
import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import math
import os

# Try to import additional libraries if available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

@dataclass
class SLAMLandmark:
    """Landmark representation for SLAM systems"""
    landmark_id: int
    class_name: str
    position: List[float]  # [x, y, z] in world coordinates
    covariance: List[List[float]]  # 3x3 covariance matrix
    observations: int  # Number of times observed
    last_seen: float  # Timestamp of last observation
    first_seen: float  # Timestamp of first observation
    confidence: float  # Overall confidence in landmark
    is_dynamic: bool = False  # Whether this landmark moves
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def update_position(self, new_position: List[float], measurement_noise: float = 0.1):
        """Update landmark position using simple Kalman filter"""
        if self.observations == 0:
            self.position = new_position.copy()
            self.covariance = [[measurement_noise**2, 0, 0],
                             [0, measurement_noise**2, 0],
                             [0, 0, measurement_noise**2]]
        else:
            # Simple update (in real SLAM, use proper Kalman filter)
            alpha = 0.1  # Learning rate
            for i in range(3):
                self.position[i] = (1 - alpha) * self.position[i] + alpha * new_position[i]
        
        self.observations += 1
        self.last_seen = time.time()

@dataclass
class CameraCalibration:
    """Camera calibration parameters"""
    fx: float = 525.0  # Focal length x
    fy: float = 525.0  # Focal length y
    cx: float = 320.0  # Principal point x
    cy: float = 240.0  # Principal point y
    width: int = 640   # Image width
    height: int = 480  # Image height
    
    # Distortion parameters (assuming minimal distortion)
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    
    def to_matrix(self) -> np.ndarray:
        """Return camera intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

class CoordinateTransformer:
    """Handles coordinate transformations for SLAM integration"""
    
    def __init__(self, camera_calibration: CameraCalibration):
        self.camera_calibration = camera_calibration
        self.camera_matrix = camera_calibration.to_matrix()
        
    def image_to_camera_coordinates(self, image_point: Tuple[float, float], depth: float) -> List[float]:
        """
        Convert image coordinates to camera coordinates
        image_point: (x, y) in image coordinates
        depth: depth in meters
        Returns: [x, y, z] in camera coordinates
        """
        if depth <= 0:
            return [0.0, 0.0, 0.0]
        
        # Convert image coordinates to normalized coordinates
        x_norm = (image_point[0] - self.camera_calibration.cx) / self.camera_calibration.fx
        y_norm = (image_point[1] - self.camera_calibration.cy) / self.camera_calibration.fy
        
        # Convert to 3D camera coordinates
        x_cam = x_norm * depth
        y_cam = y_norm * depth
        z_cam = depth
        
        return [x_cam, y_cam, z_cam]
    
    def camera_to_world_coordinates(self, camera_point: List[float], camera_pose: List[float]) -> List[float]:
        """
        Transform camera coordinates to world coordinates
        camera_point: [x, y, z] in camera coordinates
        camera_pose: [x, y, z, roll, pitch, yaw] camera pose in world coordinates
        Returns: [x, y, z] in world coordinates
        """
        if len(camera_pose) != 6:
            return camera_point  # Return as-is if pose not available
        
        # Extract position and orientation
        pos = camera_pose[:3]
        rpy = camera_pose[3:]  # roll, pitch, yaw
        
        # Create rotation matrix from roll, pitch, yaw
        R = self._euler_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
        
        # Transform point
        camera_point_np = np.array(camera_point)
        world_point = R @ camera_point_np + np.array(pos)
        
        return world_point.tolist()
    
    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        # Rotation around x-axis (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Rotation around y-axis (pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Rotation around z-axis (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        return R

class SLAMProcessor:
    """Main SLAM integration processor"""
    
    def __init__(self, camera_calibration: Optional[CameraCalibration] = None):
        self.camera_calibration = camera_calibration or CameraCalibration()
        self.coordinate_transformer = CoordinateTransformer(self.camera_calibration)
        
        # Landmark management
        self.landmarks: Dict[int, SLAMLandmark] = {}
        self.next_landmark_id = 1
        self.landmark_tracking: Dict[str, int] = {}  # class_name -> landmark_id mapping
        
        # Configuration
        self.max_landmark_distance = 10.0  # meters
        self.min_observations_for_landmark = 3
        self.landmark_timeout = 30.0  # seconds
        self.association_threshold = 1.0  # meters for data association
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'landmarks_created': 0,
            'landmarks_updated': 0,
            'objects_filtered': 0,
            'start_time': time.time()
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # History for tracking
        self.detection_history = deque(maxlen=100)
        
    def process_detection_frame(self, detection_data: List[Dict], camera_pose: Optional[List[float]] = None) -> Dict:
        """
        Process a frame of detection data for SLAM integration
        detection_data: List of detection dictionaries
        camera_pose: Current camera pose [x, y, z, roll, pitch, yaw] or None
        Returns: Processing results with landmarks and statistics
        """
        with self.lock:
            self.stats['frames_processed'] += 1
            
            # Filter detections for SLAM use
            filtered_detections = self._filter_detections_for_slam(detection_data)
            
            # Convert detections to world coordinates
            world_detections = []
            for detection in filtered_detections:
                world_detection = self._convert_detection_to_world(detection, camera_pose)
                if world_detection:
                    world_detections.append(world_detection)
            
            # Update landmarks
            self._update_landmarks(world_detections)
            
            # Clean up old landmarks
            self._cleanup_old_landmarks()
            
            # Store in history
            self.detection_history.append({
                'timestamp': time.time(),
                'detections': world_detections,
                'camera_pose': camera_pose
            })
            
            # Return results
            return {
                'timestamp': time.time(),
                'landmarks': self._get_active_landmarks(),
                'detections': world_detections,
                'statistics': self.get_statistics()
            }
    
    def _filter_detections_for_slam(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections suitable for SLAM landmarks"""
        filtered = []
        
        for detection in detections:
            # Check basic requirements
            if (detection.get('confidence', 0) < 0.5 or 
                detection.get('depth_m', 0) <= 0 or
                detection.get('depth_m', 0) > self.max_landmark_distance):
                self.stats['objects_filtered'] += 1
                continue
            
            # Filter by class - prefer static objects for SLAM
            class_name = detection.get('class_name', '')
            static_classes = ['chair', 'table', 'monitor', 'tv', 'laptop', 'book', 
                            'bottle', 'cup', 'bowl', 'plant', 'clock', 'vase']
            
            if class_name.lower() in static_classes:
                filtered.append(detection)
            elif detection.get('confidence', 0) > 0.8:  # High confidence dynamic objects
                filtered.append(detection)
                
        return filtered
    
    def _convert_detection_to_world(self, detection: Dict, camera_pose: Optional[List[float]]) -> Optional[Dict]:
        """Convert detection to world coordinates"""
        try:
            # Get image coordinates (center of bounding box)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            image_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Get depth
            depth_m = detection.get('depth_m', 0.0)
            if depth_m <= 0:
                return None
            
            # Convert to camera coordinates
            camera_coords = self.coordinate_transformer.image_to_camera_coordinates(
                image_center, depth_m
            )
            
            # Convert to world coordinates if camera pose is available
            if camera_pose:
                world_coords = self.coordinate_transformer.camera_to_world_coordinates(
                    camera_coords, camera_pose
                )
            else:
                world_coords = camera_coords
            
            # Create world detection
            world_detection = detection.copy()
            world_detection.update({
                'world_position': world_coords,
                'camera_position': camera_coords,
                'image_center': image_center,
                'camera_pose': camera_pose
            })
            
            return world_detection
            
        except Exception as e:
            print(f"Error converting detection to world coordinates: {e}")
            return None
    
    def _update_landmarks(self, world_detections: List[Dict]):
        """Update landmark map with new detections"""
        for detection in world_detections:
            world_pos = detection['world_position']
            class_name = detection.get('class_name', 'unknown')
            
            # Find existing landmark to associate with
            associated_landmark = self._find_associated_landmark(world_pos, class_name)
            
            if associated_landmark:
                # Update existing landmark
                associated_landmark.update_position(world_pos)
                associated_landmark.confidence = min(1.0, associated_landmark.confidence + 0.1)
                self.stats['landmarks_updated'] += 1
            else:
                # Create new landmark
                landmark = SLAMLandmark(
                    landmark_id=self.next_landmark_id,
                    class_name=class_name,
                    position=world_pos,
                    covariance=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                    observations=1,
                    last_seen=time.time(),
                    first_seen=time.time(),
                    confidence=detection.get('confidence', 0.5)
                )
                
                self.landmarks[self.next_landmark_id] = landmark
                self.next_landmark_id += 1
                self.stats['landmarks_created'] += 1
    
    def _find_associated_landmark(self, position: List[float], class_name: str) -> Optional[SLAMLandmark]:
        """Find existing landmark to associate with detection"""
        best_landmark = None
        best_distance = float('inf')
        
        for landmark in self.landmarks.values():
            # Check class compatibility
            if landmark.class_name != class_name:
                continue
            
            # Calculate distance
            distance = np.linalg.norm(np.array(position) - np.array(landmark.position))
            
            if distance < self.association_threshold and distance < best_distance:
                best_distance = distance
                best_landmark = landmark
        
        return best_landmark
    
    def _cleanup_old_landmarks(self):
        """Remove old landmarks that haven't been seen recently"""
        current_time = time.time()
        to_remove = []
        
        for landmark_id, landmark in self.landmarks.items():
            if (current_time - landmark.last_seen > self.landmark_timeout or
                landmark.observations < self.min_observations_for_landmark):
                to_remove.append(landmark_id)
        
        for landmark_id in to_remove:
            del self.landmarks[landmark_id]
    
    def _get_active_landmarks(self) -> List[Dict]:
        """Get currently active landmarks"""
        active_landmarks = []
        
        for landmark in self.landmarks.values():
            if landmark.observations >= self.min_observations_for_landmark:
                active_landmarks.append(landmark.to_dict())
        
        return active_landmarks
    
    def get_landmarks_for_navigation(self) -> List[Dict]:
        """Get landmarks formatted for navigation system"""
        navigation_landmarks = []
        
        for landmark in self.landmarks.values():
            if (landmark.observations >= self.min_observations_for_landmark and
                landmark.confidence > 0.5):
                
                nav_landmark = {
                    'id': landmark.landmark_id,
                    'type': landmark.class_name,
                    'position': landmark.position,
                    'uncertainty': np.trace(landmark.covariance),  # Scalar uncertainty
                    'confidence': landmark.confidence,
                    'is_static': not landmark.is_dynamic
                }
                navigation_landmarks.append(nav_landmark)
        
        return navigation_landmarks
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        runtime = time.time() - self.stats['start_time']
        return {
            **self.stats,
            'active_landmarks': len(self._get_active_landmarks()),
            'total_landmarks': len(self.landmarks),
            'runtime_seconds': runtime,
            'frames_per_second': self.stats['frames_processed'] / runtime if runtime > 0 else 0
        }
    
    def export_landmarks(self, filename: str = "slam_landmarks.json") -> bool:
        """Export landmarks to JSON file"""
        try:
            export_data = {
                'timestamp': time.time(),
                'camera_calibration': asdict(self.camera_calibration),
                'landmarks': [landmark.to_dict() for landmark in self.landmarks.values()],
                'statistics': self.get_statistics()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✓ Landmarks exported to {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to export landmarks: {e}")
            return False
    
    def import_landmarks(self, filename: str) -> bool:
        """Import landmarks from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Import landmarks
            for landmark_data in data.get('landmarks', []):
                landmark = SLAMLandmark(**landmark_data)
                self.landmarks[landmark.landmark_id] = landmark
                self.next_landmark_id = max(self.next_landmark_id, landmark.landmark_id + 1)
            
            print(f"✓ Imported {len(data.get('landmarks', []))} landmarks from {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to import landmarks: {e}")
            return False
    
    def reset_landmarks(self):
        """Reset all landmarks"""
        with self.lock:
            self.landmarks.clear()
            self.next_landmark_id = 1
            self.detection_history.clear()
            self.stats = {
                'frames_processed': 0,
                'landmarks_created': 0,
                'landmarks_updated': 0,
                'objects_filtered': 0,
                'start_time': time.time()
            }
            print("✓ Landmarks reset")

class SLAMIntegrationManager:
    """High-level manager for SLAM integration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.slam_processor = SLAMProcessor(
            CameraCalibration(**self.config.get('camera_calibration', {}))
        )
        
        # Integration settings
        self.auto_export = self.config.get('auto_export', True)
        self.export_interval = self.config.get('export_interval', 60)  # seconds
        self.export_filename = self.config.get('export_filename', 'slam_landmarks.json')
        
        # Auto-export timer
        self.last_export = time.time()
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file"""
        default_config = {
            'camera_calibration': {
                'fx': 525.0,
                'fy': 525.0,
                'cx': 320.0,
                'cy': 240.0,
                'width': 640,
                'height': 480
            },
            'auto_export': True,
            'export_interval': 60,
            'export_filename': 'slam_landmarks.json'
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def process_detection_data(self, detection_data: List[Dict], camera_pose: Optional[List[float]] = None) -> Dict:
        """Process detection data and manage auto-export"""
        result = self.slam_processor.process_detection_frame(detection_data, camera_pose)
        
        # Auto-export if enabled
        if self.auto_export:
            current_time = time.time()
            if current_time - self.last_export > self.export_interval:
                self.slam_processor.export_landmarks(self.export_filename)
                self.last_export = current_time
        
        return result
    
    def get_navigation_data(self) -> Dict:
        """Get data formatted for navigation system"""
        return {
            'landmarks': self.slam_processor.get_landmarks_for_navigation(),
            'statistics': self.slam_processor.get_statistics(),
            'timestamp': time.time()
        }

if __name__ == "__main__":
    # Example usage
    print("SLAM Integration Test")
    
    # Create SLAM integration manager
    slam_manager = SLAMIntegrationManager()
    
    # Simulate detection data
    test_detections = [
        {
            'tracker_id': 1,
            'class_name': 'chair',
            'confidence': 0.85,
            'bbox': [100, 150, 200, 300],
            'depth_mm': 2500,
            'depth_m': 2.5
        },
        {
            'tracker_id': 2,
            'class_name': 'table',
            'confidence': 0.92,
            'bbox': [300, 100, 450, 250],
            'depth_mm': 3200,
            'depth_m': 3.2
        }
    ]
    
    # Simulate camera pose (optional)
    camera_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, z, roll, pitch, yaw]
    
    # Process detection data
    for i in range(5):
        result = slam_manager.process_detection_data(test_detections, camera_pose)
        print(f"Frame {i+1}: {result['statistics']['active_landmarks']} active landmarks")
        
        # Simulate camera movement
        camera_pose[0] += 0.1  # Move forward
        time.sleep(0.1)
    
    # Get navigation data
    nav_data = slam_manager.get_navigation_data()
    print(f"Navigation landmarks: {len(nav_data['landmarks'])}")
    
    # Export landmarks
    slam_manager.slam_processor.export_landmarks("test_landmarks.json")
    
    print("SLAM integration test completed") 