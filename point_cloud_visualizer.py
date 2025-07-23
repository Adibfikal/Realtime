"""
Point Cloud Visualizer for Real-time Object Detection with 3D Position
Creates a live text-based visualization of detected objects with their 3D positions
"""

import numpy as np
import time
from typing import List, Dict, Optional
from datetime import datetime

class PointCloudVisualizer:
    """Simple text-based visualizer for 3D object positions"""
    
    def __init__(self, max_objects=10):
        self.max_objects = max_objects
        self.object_history = {}
        self.last_update = 0
        
    def update_detections(self, detection_metadata: Dict):
        """Update visualization with new detection data"""
        if not detection_metadata or 'detections' not in detection_metadata:
            return
        
        current_time = time.time()
        self.last_update = current_time
        
        # Clear old objects (older than 2 seconds)
        self._cleanup_old_objects(current_time, timeout=2.0)
        
        # Process new detections
        for detection in detection_metadata['detections']:
            if detection['position_3d']['valid']:
                tracker_id = detection.get('tracker_id', 'unknown')
                
                self.object_history[tracker_id] = {
                    'class_name': detection['class_name'],
                    'position_3d': detection['position_3d'],
                    'confidence': detection.get('confidence', 0.0),
                    'last_seen': current_time,
                    'bbox': detection['bbox']
                }
    
    def _cleanup_old_objects(self, current_time: float, timeout: float = 2.0):
        """Remove objects that haven't been seen recently"""
        to_remove = []
        for obj_id, obj_data in self.object_history.items():
            if current_time - obj_data['last_seen'] > timeout:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.object_history[obj_id]
    
    def get_visualization_text(self) -> str:
        """Get formatted text visualization of current objects"""
        if not self.object_history:
            return "No objects detected with valid 3D positions"
        
        lines = []
        lines.append("=" * 70)
        lines.append("REAL-TIME 3D OBJECT POSITIONS")
        lines.append(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
        lines.append("=" * 70)
        lines.append(f"{'ID':<4} {'Object':<12} {'X(m)':<8} {'Y(m)':<8} {'Z(m)':<8} {'Az(°)':<7} {'El(°)':<7} {'Conf':<5}")
        lines.append("-" * 70)
        
        # Sort objects by distance (closest first)
        sorted_objects = sorted(
            self.object_history.items(), 
            key=lambda x: x[1]['position_3d']['z']
        )
        
        for obj_id, obj_data in sorted_objects[:self.max_objects]:
            pos = obj_data['position_3d']
            lines.append(
                f"{str(obj_id):<4} "
                f"{obj_data['class_name'][:12]:<12} "
                f"{pos['x']:<8.2f} "
                f"{pos['y']:<8.2f} "
                f"{pos['z']:<8.2f} "
                f"{pos['azimuth']:<7.1f} "
                f"{pos['elevation']:<7.1f} "
                f"{obj_data['confidence']:<5.2f}"
            )
        
        lines.append("=" * 70)
        lines.append(f"Total objects with 3D data: {len(self.object_history)}")
        
        return "\n".join(lines)
    
    def get_closest_object(self) -> Optional[Dict]:
        """Get the closest object to the camera"""
        if not self.object_history:
            return None
        
        closest_obj = min(
            self.object_history.items(), 
            key=lambda x: x[1]['position_3d']['z']
        )
        
        return {
            'id': closest_obj[0],
            'data': closest_obj[1]
        }
    
    def get_objects_in_range(self, min_distance: float = 0.0, max_distance: float = 10.0) -> List[Dict]:
        """Get objects within a specified distance range"""
        objects_in_range = []
        
        for obj_id, obj_data in self.object_history.items():
            distance = obj_data['position_3d']['z']
            if min_distance <= distance <= max_distance:
                objects_in_range.append({
                    'id': obj_id,
                    'data': obj_data
                })
        
        # Sort by distance
        objects_in_range.sort(key=lambda x: x['data']['position_3d']['z'])
        return objects_in_range
    
    def get_objects_by_angle(self, azimuth_range: tuple = (-45, 45), 
                           elevation_range: tuple = (-30, 30)) -> List[Dict]:
        """Get objects within specified angular ranges"""
        objects_in_angle = []
        
        for obj_id, obj_data in self.object_history.items():
            az = obj_data['position_3d']['azimuth']
            el = obj_data['position_3d']['elevation']
            
            if (azimuth_range[0] <= az <= azimuth_range[1] and 
                elevation_range[0] <= el <= elevation_range[1]):
                objects_in_angle.append({
                    'id': obj_id,
                    'data': obj_data
                })
        
        return objects_in_angle
    
    def get_statistics(self) -> Dict:
        """Get current statistics about detected objects"""
        if not self.object_history:
            return {
                'total_objects': 0,
                'avg_distance': 0.0,
                'closest_distance': 0.0,
                'farthest_distance': 0.0
            }
        
        distances = [obj['position_3d']['z'] for obj in self.object_history.values()]
        
        return {
            'total_objects': len(self.object_history),
            'avg_distance': np.mean(distances),
            'closest_distance': min(distances),
            'farthest_distance': max(distances),
            'avg_azimuth': np.mean([obj['position_3d']['azimuth'] for obj in self.object_history.values()]),
            'avg_elevation': np.mean([obj['position_3d']['elevation'] for obj in self.object_history.values()])
        }
