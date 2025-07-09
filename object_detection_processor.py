"""
Object Detection Processor for Real-time YOLO Detection with Depth Information
Integrates YOLO v11 object detection with Intel RealSense depth data
"""

import cv2
import numpy as np
import os
import time
from typing import Optional, Tuple, Dict, List
import json

try:
    from ultralytics import YOLO
    import supervision as sv
    import torch
    DETECTION_AVAILABLE = True
except ImportError as e:
    DETECTION_AVAILABLE = False
    IMPORT_ERROR = str(e)

class ObjectDetectionProcessor:
    """Handles real-time object detection with depth information"""
    
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
        
        # Configuration from notebook
        self.YOLO_INPUT_SIZE = 640
        self.CENTER_REGION_RATIO = 0.6
        self.DEPTH_OUTLIER_THRESHOLD = 2.0
        self.LABEL_FONT_SCALE = 0.8
        self.LABEL_FONT_THICKNESS = 2
        self.LABEL_PADDING = 8
        
        # Recording metadata
        self.detection_metadata = []
        
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if required dependencies are available"""
        if not DETECTION_AVAILABLE:
            return False, f"Missing dependencies: {IMPORT_ERROR}"
        return True, "All dependencies available"
    
    def load_model(self, model_path: Optional[str] = None) -> Tuple[bool, str]:
        """Load YOLO model and initialize components"""
        if not DETECTION_AVAILABLE:
            return False, "Detection dependencies not available"
        
        if model_path:
            self.model_path = model_path
            
        if not os.path.exists(self.model_path):
            return False, f"Model file not found: {self.model_path}"
        
        try:
            # Load YOLO model
            self.model = YOLO(self.model_path)
            
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
                text_color=sv.Color.WHITE
            )
            
            self.detection_stats['model_loaded'] = True
            return True, f"Model loaded successfully: {os.path.basename(self.model_path)}"
            
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"
    
    def calculate_depth_robust_center(self, depth_array: np.ndarray, bbox: List[float]) -> float:
        """
        Calculate depth using robust center method with outlier removal
        Based on the notebook implementation
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Calculate center region
        width = x2 - x1
        height = y2 - y1
        
        margin_x = int(width * (1 - self.CENTER_REGION_RATIO) / 2)
        margin_y = int(height * (1 - self.CENTER_REGION_RATIO) / 2)
        
        center_x1 = max(0, x1 + margin_x)
        center_y1 = max(0, y1 + margin_y)
        center_x2 = min(depth_array.shape[1], x2 - margin_x)
        center_y2 = min(depth_array.shape[0], y2 - margin_y)
        
        if center_x2 <= center_x1 or center_y2 <= center_y1:
            return 0.0
        
        bbox_depth = depth_array[center_y1:center_y2, center_x1:center_x2]
        valid_depth = bbox_depth[(bbox_depth > 0) & (bbox_depth < 10000)]
        
        if len(valid_depth) < 5:
            return 0.0
        
        # Remove statistical outliers
        mean_depth = np.mean(valid_depth)
        std_depth = np.std(valid_depth)
        
        # Keep only values within threshold standard deviations
        filtered_depth = valid_depth[
            np.abs(valid_depth - mean_depth) <= self.DEPTH_OUTLIER_THRESHOLD * std_depth
        ]
        
        return np.mean(filtered_depth) if len(filtered_depth) > 0 else mean_depth
    
    def process_frame(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process frame with object detection and depth calculation
        Returns annotated frame and detection info
        """
        if not self.model or not DETECTION_AVAILABLE:
            return rgb_frame, {'error': 'Model not loaded'}
        
        start_time = time.time()
        
        try:
            # Run YOLO detection
            results = self.model(rgb_frame, imgsz=self.YOLO_INPUT_SIZE, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Update tracker
            if self.tracker:
                detections = self.tracker.update_with_detections(detections)
            
            # Calculate depths for each detection
            depth_values = []
            detection_info = []
            
            if len(detections.xyxy) > 0:
                for i, bbox in enumerate(detections.xyxy):
                    avg_depth = self.calculate_depth_robust_center(depth_frame, bbox)
                    depth_values.append(avg_depth)
                    
                    # Store detection info for metadata
                    if hasattr(detections, 'class_id') and hasattr(detections, 'tracker_id'):
                        class_id = detections.class_id[i] if i < len(detections.class_id) else None
                        tracker_id = detections.tracker_id[i] if i < len(detections.tracker_id) else None
                        confidence = detections.confidence[i] if hasattr(detections, 'confidence') and i < len(detections.confidence) else None
                        class_name = results.names[class_id] if class_id is not None and hasattr(results, 'names') else "unknown"
                        
                        detection_info.append({
                            'tracker_id': int(tracker_id) if tracker_id is not None else None,
                            'class_id': int(class_id) if class_id is not None else None,
                            'class_name': class_name,
                            'confidence': float(confidence) if confidence is not None else None,
                            'bbox': [float(x) for x in bbox],
                            'depth_mm': float(avg_depth),
                            'depth_m': float(avg_depth / 1000.0) if avg_depth > 0 else 0.0
                        })
            
            # Create labels for annotation
            labels = []
            if hasattr(detections, 'class_id') and hasattr(detections, 'tracker_id'):
                for i, (class_id, tracker_id) in enumerate(zip(detections.class_id, detections.tracker_id)):
                    class_name = results.names[class_id] if class_id is not None and hasattr(results, 'names') else "unknown"
                    
                    if i < len(depth_values) and depth_values[i] > 0:
                        depth_m = depth_values[i] / 1000.0
                        label = f"#{tracker_id} {class_name} ({depth_m:.2f}m)"
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
            
            # Store metadata for recording
            frame_metadata = {
                'frame_number': self.frame_count,
                'timestamp': time.time(),
                'processing_time_ms': processing_time,
                'detections': detection_info
            }
            
            self.frame_count += 1
            
            return annotated_frame, frame_metadata
            
        except Exception as e:
            return rgb_frame, {'error': f'Processing failed: {str(e)}'}
    
    def get_statistics(self) -> Dict:
        """Get current detection statistics"""
        return self.detection_stats.copy()
    
    def add_detection_metadata(self, metadata: Dict):
        """Add frame metadata for recording"""
        self.detection_metadata.append(metadata)
    
    def save_detection_metadata(self, output_path: str) -> bool:
        """Save detection metadata to JSON file"""
        try:
            metadata_file = output_path.replace('.avi', '_detections.json').replace('.mp4', '_detections.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'total_frames': len(self.detection_metadata),
                    'model_path': self.model_path,
                    'config': {
                        'center_region_ratio': self.CENTER_REGION_RATIO,
                        'depth_outlier_threshold': self.DEPTH_OUTLIER_THRESHOLD,
                        'yolo_input_size': self.YOLO_INPUT_SIZE
                    },
                    'frames': self.detection_metadata
                }, f, indent=2)
            print(f"Detection metadata saved: {metadata_file}")
            return True
        except Exception as e:
            print(f"Failed to save metadata: {e}")
            return False
    
    def clear_metadata(self):
        """Clear stored detection metadata"""
        self.detection_metadata.clear()
        self.frame_count = 0 