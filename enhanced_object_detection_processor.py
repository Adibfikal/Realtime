"""
Enhanced Object Detection Processor for Real-time YOLO Detection with Advanced 3D Analysis
Integrates YOLO v11 object detection with Intel RealSense depth data, weighted center point extraction, and angle calculations
"""

import cv2
import numpy as np
import os
import time
import math
from typing import Optional, Tuple, Dict, List
import json

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    from ultralytics import YOLO
    import supervision as sv
    import torch
    DETECTION_AVAILABLE = True
except ImportError as e:
    DETECTION_AVAILABLE = False
    IMPORT_ERROR = str(e)

class EnhancedObjectDetectionProcessor:
    """Enhanced processor with weighted center point extraction and angle calculations"""
    
    def __init__(self, model_path: str = "PredictModel/best PT 3500.pt"):
        self.model_path = model_path
        self.model = None
        self.tracker = None
        self.box_annotator = None
        self.label_annotator = None
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.detection_stats = {
            'objects_detected': 0,
            'avg_processing_time': 0.0,
            'model_loaded': False,
            'last_detection_count': 0
        }
        
        # Configuration from plan
        self.YOLO_INPUT_SIZE = 640
        self.CENTER_REGION_RATIO = 0.6
        self.DEPTH_OUTLIER_THRESHOLD = 2.0
        self.LABEL_FONT_SCALE = 0.8
        self.LABEL_FONT_THICKNESS = 2
        self.LABEL_PADDING = 8
        
        # Recording metadata
        self.detection_metadata = []
        
        # Enhanced configuration for point cloud processing
        self.POINT_CLOUD_CONFIG = {
            'weighting_method': 'combined',
            'distance_weight': 0.3,
            'validity_weight': 0.3,
            'statistical_weight': 0.2,
            'spatial_weight': 0.2,
            'min_valid_points': 10,
            'max_processing_region': 0.8
        }
        
        # Angle calculation configuration
        self.ANGLE_CONFIG = {
            'coordinate_system': 'camera_centered',
            'angle_units': 'degrees',
            'precision': 1
        }
        
        # Display options configuration
        self.DISPLAY_OPTIONS = {
            'show_coordinates': True,
            'show_angles': True,
            'coordinate_precision': 2,
            'angle_precision': 1,
            'compact_display': False
        }
        
        # Camera intrinsics (will be updated with actual values)
        self.camera_intrinsics = {
            'fx': 525.0,  # Default focal length x
            'fy': 525.0,  # Default focal length y
            'cx': 320.0,  # Default principal point x
            'cy': 240.0,  # Default principal point y
            'width': 640,  # Image width
            'height': 480  # Image height
        }
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if required dependencies are available"""
        if not DETECTION_AVAILABLE:
            return False, f"Missing dependencies: {IMPORT_ERROR}"
        return True, "All dependencies available"
    
    def get_camera_intrinsics(self, pipeline: Optional['rs.pipeline'] = None) -> Dict:
        """
        Get camera intrinsic parameters from RealSense camera
        Returns dictionary with fx, fy, cx, cy, width, height
        """
        if not REALSENSE_AVAILABLE or not pipeline:
            print("⚠ Using default camera intrinsics - RealSense not available or no pipeline provided")
            return self.camera_intrinsics
        
        try:
            # Get the active profile
            profile = pipeline.get_active_profile()
            
            # Get the color stream profile
            color_profile = profile.get_stream(rs.stream.color)
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            # Update camera intrinsics with actual values
            self.camera_intrinsics.update({
                'fx': color_intrinsics.fx,
                'fy': color_intrinsics.fy,
                'cx': color_intrinsics.ppx,
                'cy': color_intrinsics.ppy,
                'width': color_intrinsics.width,
                'height': color_intrinsics.height
            })
            
            print(f"✓ Camera intrinsics loaded: fx={color_intrinsics.fx:.1f}, fy={color_intrinsics.fy:.1f}")
            return self.camera_intrinsics
            
        except Exception as e:
            print(f"⚠ Failed to get camera intrinsics, using defaults: {e}")
            return self.camera_intrinsics
    
    def calculate_weighted_center_point(self, depth_array: np.ndarray, bbox: List[float]) -> Tuple[Tuple[int, int], float, Dict]:
        """
        Calculate weighted center point from object's point cloud
        
        Args:
            depth_array: Depth image array
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Tuple containing:
            - (center_x, center_y): Image coordinates of weighted center
            - depth_value: Depth at center point in mm
            - metadata: Additional information about the calculation
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Calculate processing region (smaller than full bbox for better accuracy)
        width = x2 - x1
        height = y2 - y1
        
        margin_ratio = (1 - self.POINT_CLOUD_CONFIG['max_processing_region']) / 2
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        
        region_x1 = max(0, x1 + margin_x)
        region_y1 = max(0, y1 + margin_y)
        region_x2 = min(depth_array.shape[1], x2 - margin_x)
        region_y2 = min(depth_array.shape[0], y2 - margin_y)
        
        if region_x2 <= region_x1 or region_y2 <= region_y1:
            return (int((x1 + x2) / 2), int((y1 + y2) / 2)), 0.0, {'error': 'invalid_region'}
        
        # Extract depth region
        depth_region = depth_array[region_y1:region_y2, region_x1:region_x2]
        valid_mask = (depth_region > 0) & (depth_region < 10000)
        
        if np.sum(valid_mask) < self.POINT_CLOUD_CONFIG['min_valid_points']:
            return (int((x1 + x2) / 2), int((y1 + y2) / 2)), 0.0, {'error': 'insufficient_points'}
        
        valid_depths = depth_region[valid_mask]
        
        # Calculate statistical properties
        depth_mean = np.mean(valid_depths)
        depth_std = np.std(valid_depths)
        depth_median = np.median(valid_depths)
        
        # Remove statistical outliers
        outlier_mask = np.abs(valid_depths - depth_mean) <= (self.DEPTH_OUTLIER_THRESHOLD * depth_std)
        filtered_depths = valid_depths[outlier_mask]
        
        if len(filtered_depths) < self.POINT_CLOUD_CONFIG['min_valid_points']:
            filtered_depths = valid_depths  # Use all valid points if filtering removes too many
        
        # Create coordinate grids for weighting
        y_coords, x_coords = np.meshgrid(
            np.arange(region_y1, region_y2),
            np.arange(region_x1, region_x2),
            indexing='ij'
        )
        
        # Calculate weights for each valid point
        weights = np.zeros_like(depth_region, dtype=np.float32)
        
        for i in range(depth_region.shape[0]):
            for j in range(depth_region.shape[1]):
                if not valid_mask[i, j]:
                    continue
                
                depth = depth_region[i, j]
                
                # 1. Distance weight (closer points get higher weight)
                distance_weight = 1.0 / (1.0 + depth / 1000.0)  # Convert mm to m
                distance_weight *= self.POINT_CLOUD_CONFIG['distance_weight']
                
                # 2. Validity weight (points with consistent neighbors)
                validity_weight = self._calculate_validity_weight(depth_region, i, j, valid_mask)
                validity_weight *= self.POINT_CLOUD_CONFIG['validity_weight']
                
                # 3. Statistical weight (points closer to median)
                stat_diff = abs(depth - depth_median) / depth_std if depth_std > 0 else 0
                statistical_weight = np.exp(-stat_diff) * self.POINT_CLOUD_CONFIG['statistical_weight']
                
                # 4. Spatial weight (points closer to geometric center)
                geo_center_x = (x1 + x2) / 2
                geo_center_y = (y1 + y2) / 2
                spatial_dist = np.sqrt((x_coords[i, j] - geo_center_x)**2 + (y_coords[i, j] - geo_center_y)**2)
                max_spatial_dist = np.sqrt(width**2 + height**2) / 2
                spatial_weight = (1.0 - spatial_dist / max_spatial_dist) * self.POINT_CLOUD_CONFIG['spatial_weight']
                
                # Combine weights
                weights[i, j] = distance_weight + validity_weight + statistical_weight + spatial_weight
        
        # Calculate weighted center
        if np.sum(weights) == 0:
            # Fallback to geometric center
            weighted_center_x = int((x1 + x2) / 2)
            weighted_center_y = int((y1 + y2) / 2)
            weighted_depth = float(depth_mean)
        else:
            total_weight = np.sum(weights[valid_mask])
            
            # Calculate weighted coordinates
            weighted_x = np.sum(x_coords[valid_mask] * weights[valid_mask]) / total_weight
            weighted_y = np.sum(y_coords[valid_mask] * weights[valid_mask]) / total_weight
            weighted_depth = np.sum(depth_region[valid_mask] * weights[valid_mask]) / total_weight
            
            weighted_center_x = int(weighted_x)
            weighted_center_y = int(weighted_y)
        
        # Calculate quality metrics
        point_confidence = min(1.0, len(filtered_depths) / 50.0)  # Confidence based on number of points
        depth_quality = 1.0 / (1.0 + depth_std / depth_mean) if depth_mean > 0 else 0.0
        
        metadata = {
            'point_confidence': point_confidence,
            'depth_quality': depth_quality,
            'valid_points': int(np.sum(valid_mask)),
            'filtered_points': len(filtered_depths),
            'depth_statistics': {
                'mean': float(depth_mean),
                'std': float(depth_std),
                'median': float(depth_median),
                'min': float(np.min(valid_depths)),
                'max': float(np.max(valid_depths))
            }
        }
        
        return (weighted_center_x, weighted_center_y), float(weighted_depth), metadata
    
    def _calculate_validity_weight(self, depth_region: np.ndarray, i: int, j: int, valid_mask: np.ndarray) -> float:
        """Calculate validity weight based on neighborhood consistency"""
        window_size = 3
        half_window = window_size // 2
        
        # Get neighborhood bounds
        i_start = max(0, i - half_window)
        i_end = min(depth_region.shape[0], i + half_window + 1)
        j_start = max(0, j - half_window)
        j_end = min(depth_region.shape[1], j + half_window + 1)
        
        # Extract neighborhood
        neighborhood = depth_region[i_start:i_end, j_start:j_end]
        neighborhood_mask = valid_mask[i_start:i_end, j_start:j_end]
        
        if np.sum(neighborhood_mask) < 2:
            return 0.5  # Medium confidence for isolated points
        
        center_depth = depth_region[i, j]
        neighbor_depths = neighborhood[neighborhood_mask]
        
        # Calculate consistency (lower standard deviation = higher consistency)
        depth_std = np.std(neighbor_depths)
        consistency = np.exp(-depth_std / center_depth) if center_depth > 0 else 0.0
        
        return consistency
    
    def image_to_3d_coordinates(self, image_point: Tuple[int, int], depth_mm: float, intrinsics: Dict) -> Tuple[float, float, float]:
        """
        Convert image coordinates to 3D camera coordinates using proper intrinsics
        
        Args:
            image_point: (x, y) in image coordinates
            depth_mm: Depth in millimeters
            intrinsics: Camera intrinsic parameters
        
        Returns:
            (x, y, z) in meters relative to camera center
        """
        if depth_mm <= 0:
            return (0.0, 0.0, 0.0)
        
        # Convert depth to meters
        depth_m = depth_mm / 1000.0
        
        # Extract image coordinates
        u, v = image_point
        
        # Convert to normalized coordinates using camera intrinsics
        x_norm = (u - intrinsics['cx']) / intrinsics['fx']
        y_norm = (v - intrinsics['cy']) / intrinsics['fy']
        
        # Convert to 3D camera coordinates
        x_cam = x_norm * depth_m
        y_cam = y_norm * depth_m
        z_cam = depth_m
        
        return (x_cam, y_cam, z_cam)
    
    def calculate_spherical_angles(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """
        Calculate azimuth and elevation angles from 3D coordinates
        
        Args:
            x, y, z: 3D coordinates in meters
        
        Returns:
            (azimuth, elevation) in degrees
            - azimuth: Horizontal angle from camera optical axis (-180° to +180°)
            - elevation: Vertical angle from horizontal plane (-90° to +90°)
        """
        if z == 0:
            return (0.0, 0.0)
        
        # Calculate azimuth (horizontal angle from optical axis)
        azimuth_rad = math.atan2(x, z)
        azimuth_deg = math.degrees(azimuth_rad)
        
        # Calculate elevation (vertical angle from horizontal plane)
        horizontal_distance = math.sqrt(x**2 + z**2)
        elevation_rad = math.atan2(y, horizontal_distance) if horizontal_distance > 0 else 0.0
        elevation_deg = math.degrees(elevation_rad)
        
        return (azimuth_deg, elevation_deg)
    
    def load_model(self, model_path: Optional[str] = None) -> Tuple[bool, str]:
        """Load YOLO model and initialize components"""
        if not DETECTION_AVAILABLE:
            return False, "Detection dependencies not available"
        
        if model_path:
            self.model_path = model_path
            
        if not os.path.exists(self.model_path):
            return False, f"Model file not found: {self.model_path}"
        
        try:
            # Load YOLO model with proper handling for PyTorch 2.6+ weights_only behavior
            import torch
            
            # Check ultralytics version and handle compatibility
            try:
                import ultralytics
                from ultralytics.nn.modules import block
                
                print(f"Ultralytics version: {ultralytics.__version__}")
                
                # Check if required modules exist for the model
                missing_modules = []
                if not hasattr(block, 'C3k2'):
                    missing_modules.append('C3k2')
                
                if missing_modules:
                    print(f"⚠ Missing modules in current ultralytics version: {missing_modules}")
                    print("This model requires a newer version of ultralytics")
                    print("Please run: pip install ultralytics>=8.2.0")
                    
                    # Try compatibility workarounds
                    for module_name in missing_modules:
                        if module_name == 'C3k2':
                            if hasattr(block, 'C2f'):
                                setattr(block, 'C3k2', getattr(block, 'C2f'))
                                print(f"✓ Added {module_name} compatibility using C2f")
                            elif hasattr(block, 'C3'):
                                setattr(block, 'C3k2', getattr(block, 'C3'))
                                print(f"✓ Added {module_name} compatibility using C3")
                            else:
                                print(f"❌ Cannot create compatibility for {module_name}")
                                
            except Exception as compat_error:
                print(f"⚠ Version compatibility check failed: {compat_error}")
            
            # Temporarily set torch.load to use weights_only=False for trusted model loading
            # This is necessary for custom trained YOLO models with PyTorch 2.6+
            original_load = torch.load
            def safe_load_for_yolo(*args, **kwargs):
                # Force weights_only=False for model loading (trusted source)
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            # Patch torch.load temporarily
            torch.load = safe_load_for_yolo
            
            try:
                self.model = YOLO(self.model_path)
                print("✓ Enhanced YOLO model loaded successfully")
            finally:
                # Always restore original torch.load
                torch.load = original_load
            
            # Performance optimizations
            self.model.overrides['imgsz'] = self.YOLO_INPUT_SIZE
            
            # Check for GPU and use half precision if available
            if torch.cuda.is_available():
                try:
                    self.model.to('cuda')
                    self.model.half()
                    print("Using GPU with half precision")
                except:
                    print("GPU available but half precision failed, using full precision")
            
            # Warmup the model
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False)
            
            # Initialize tracker and annotators
            self.tracker = sv.ByteTrack()
            self.box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator(
                text_scale=self.LABEL_FONT_SCALE,
                text_thickness=self.LABEL_FONT_THICKNESS,
                text_padding=self.LABEL_PADDING,
                text_color=sv.Color.white()
            )
            
            self.detection_stats['model_loaded'] = True
            return True, f"Enhanced model loaded successfully: {os.path.basename(self.model_path)}"
            
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
    
    def process_frame(self, rgb_frame: np.ndarray, depth_frame: np.ndarray, pipeline: Optional['rs.pipeline'] = None) -> Tuple[np.ndarray, Dict]:
        """
        Process frame with enhanced object detection, weighted center point extraction, and angle calculation
        Returns annotated frame and enhanced detection info
        """
        if not self.model or not DETECTION_AVAILABLE:
            return rgb_frame, {'error': 'Model not loaded'}
        
        start_time = time.time()
        
        # Update camera intrinsics if pipeline is provided
        if pipeline and self.frame_count % 30 == 0:  # Update every 30 frames
            self.get_camera_intrinsics(pipeline)
        
        try:
            # Run YOLO detection
            results = self.model(rgb_frame, imgsz=self.YOLO_INPUT_SIZE, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Update tracker
            if self.tracker:
                detections = self.tracker.update_with_detections(detections)
            
            # Process each detection with enhanced analysis
            enhanced_detection_info = []
            labels = []
            
            if len(detections.xyxy) > 0:
                for i, bbox in enumerate(detections.xyxy):
                    # Extract weighted center point and depth
                    center_point, depth_value, point_metadata = self.calculate_weighted_center_point(depth_frame, bbox)
                    
                    # Convert to 3D coordinates
                    center_3d = self.image_to_3d_coordinates(center_point, depth_value, self.camera_intrinsics)
                    
                    # Calculate spherical angles
                    azimuth, elevation = self.calculate_spherical_angles(*center_3d)
                    
                    # Gather detection metadata
                    if hasattr(detections, 'class_id') and hasattr(detections, 'tracker_id'):
                        class_id = detections.class_id[i] if detections.class_id is not None and i < len(detections.class_id) else None
                        tracker_id = detections.tracker_id[i] if detections.tracker_id is not None and i < len(detections.tracker_id) else None
                        confidence = detections.confidence[i] if hasattr(detections, 'confidence') and detections.confidence is not None and i < len(detections.confidence) else None
                        class_name = results.names[class_id] if class_id is not None and hasattr(results, 'names') else "unknown"
                        
                        # Enhanced detection info with all new fields
                        detection_info = {
                            # Existing fields
                            'tracker_id': int(tracker_id) if tracker_id is not None else None,
                            'class_id': int(class_id) if class_id is not None else None,
                            'class_name': class_name,
                            'confidence': float(confidence) if confidence is not None else None,
                            'bbox': [float(x) for x in bbox],
                            'depth_mm': float(depth_value),
                            'depth_m': float(depth_value / 1000.0) if depth_value > 0 else 0.0,
                            
                            # New enhanced fields
                            'center_point_image': [int(center_point[0]), int(center_point[1])],
                            'center_point_3d': [float(center_3d[0]), float(center_3d[1]), float(center_3d[2])],
                            'azimuth_deg': float(azimuth),
                            'elevation_deg': float(elevation),
                            'point_confidence': point_metadata.get('point_confidence', 0.0),
                            'depth_quality': point_metadata.get('depth_quality', 0.0),
                            'depth_statistics': point_metadata.get('depth_statistics', {}),
                            'camera_intrinsics': self.camera_intrinsics.copy()
                        }
                        
                        enhanced_detection_info.append(detection_info)
                        
                        # Create enhanced labels
                        if depth_value > 0:
                            if self.DISPLAY_OPTIONS['compact_display']:
                                label = f"#{tracker_id} {class_name} ({center_3d[2]:.{self.DISPLAY_OPTIONS['coordinate_precision']}f}m)"
                            else:
                                x, y, z = center_3d
                                label = (f"#{tracker_id} {class_name}\n"
                                        f"({x:.{self.DISPLAY_OPTIONS['coordinate_precision']}f}, "
                                        f"{y:.{self.DISPLAY_OPTIONS['coordinate_precision']}f}, "
                                        f"{z:.{self.DISPLAY_OPTIONS['coordinate_precision']}f})m\n"
                                        f"[Az:{azimuth:.{self.DISPLAY_OPTIONS['angle_precision']}f}°, "
                                        f"El:{elevation:.{self.DISPLAY_OPTIONS['angle_precision']}f}°]")
                        else:
                            label = f"#{tracker_id} {class_name} (No depth)"
                        
                        labels.append(label)
            
            # Annotate frame
            annotated_frame = rgb_frame.copy()
            if self.box_annotator:
                annotated_frame = self.box_annotator.annotate(annotated_frame, detections=detections)
            if self.label_annotator:
                annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 30:  # Keep last 30 measurements
                self.processing_times.pop(0)
            
            self.detection_stats.update({
                'objects_detected': len(detections.xyxy),
                'avg_processing_time': np.mean(self.processing_times),
                'last_detection_count': len(detections.xyxy)
            })
            
            # Enhanced frame metadata
            frame_metadata = {
                'frame_number': self.frame_count,
                'timestamp': time.time(),
                'processing_time_ms': processing_time,
                'detections': enhanced_detection_info,
                'camera_intrinsics': self.camera_intrinsics.copy(),
                'configuration': {
                    'point_cloud_config': self.POINT_CLOUD_CONFIG,
                    'angle_config': self.ANGLE_CONFIG,
                    'display_options': self.DISPLAY_OPTIONS
                }
            }
            
            self.frame_count += 1
            
            return annotated_frame, frame_metadata
            
        except Exception as e:
            return rgb_frame, {'error': f'Enhanced processing failed: {str(e)}'}
    
    def get_statistics(self) -> Dict:
        """Get current detection statistics"""
        stats = self.detection_stats.copy()
        stats.update({
            'enhanced_features': True,
            'camera_intrinsics_loaded': bool(self.camera_intrinsics),
            'point_cloud_processing': True,
            'angle_calculation': True
        })
        return stats
    
    def add_detection_metadata(self, metadata: Dict):
        """Add frame metadata for recording"""
        self.detection_metadata.append(metadata)
    
    def save_detection_metadata(self, output_path: str) -> bool:
        """Save enhanced detection metadata to JSON file"""
        try:
            metadata_file = output_path.replace('.avi', '_enhanced_detections.json').replace('.mp4', '_enhanced_detections.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'version': '2.0_enhanced',
                    'total_frames': len(self.detection_metadata),
                    'model_path': self.model_path,
                    'camera_intrinsics': self.camera_intrinsics,
                    'config': {
                        'center_region_ratio': self.CENTER_REGION_RATIO,
                        'depth_outlier_threshold': self.DEPTH_OUTLIER_THRESHOLD,
                        'yolo_input_size': self.YOLO_INPUT_SIZE,
                        'point_cloud_config': self.POINT_CLOUD_CONFIG,
                        'angle_config': self.ANGLE_CONFIG,
                        'display_options': self.DISPLAY_OPTIONS
                    },
                    'frames': self.detection_metadata
                }, f, indent=2)
            print(f"Enhanced detection metadata saved: {metadata_file}")
            return True
        except Exception as e:
            print(f"Failed to save enhanced metadata: {e}")
            return False
    
    def clear_metadata(self):
        """Clear stored detection metadata"""
        self.detection_metadata.clear()
        self.frame_count = 0
    
    def update_configuration(self, config_updates: Dict):
        """Update configuration parameters"""
        if 'point_cloud_processing' in config_updates:
            self.POINT_CLOUD_CONFIG.update(config_updates['point_cloud_processing'])
        
        if 'angle_calculation' in config_updates:
            self.ANGLE_CONFIG.update(config_updates['angle_calculation'])
        
        if 'visualization' in config_updates:
            self.DISPLAY_OPTIONS.update(config_updates['visualization'])
        
        print("✓ Configuration updated")
