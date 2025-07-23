"""
Integration Example: Object Detection + SLAM + Communication
Demonstrates how to integrate object detection with SLAM navigation system
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict, List
import json
import argparse

# Import our modules
from object_detection_processor import ObjectDetectionProcessor
from communication_handler import CommunicationHandler, create_communication_handler
from slam_integration import SLAMIntegrationManager

# Try to import RealSense if available
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

class IntegratedSystem:
    """
    Main system that integrates object detection, SLAM processing, and communication
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.object_detector = ObjectDetectionProcessor(
            self.config.get('model_path', 'PredictModel/best PT 3500.pt')
        )
        
        self.slam_manager = SLAMIntegrationManager(config_file)
        
        # Communication handler will be initialized based on config
        self.communication_handler = None
        
        # Camera components
        self.camera_pipeline = None
        self.camera_running = False
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'detections_sent': 0,
            'landmarks_created': 0,
            'start_time': time.time()
        }
        
        # Current camera pose (would be updated by navigation system)
        self.camera_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, z, roll, pitch, yaw]
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Status flags
        self.running = False
        
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load system configuration"""
        default_config = {
            'model_path': 'PredictModel/best PT 3500.pt',
            'communication': {
                'protocol': 'socket',
                'host': 'localhost',
                'port': 8888
            },
            'camera': {
                'width': 640,
                'height': 480,
                'fps': 30,
                'use_realsense': True
            },
            'processing': {
                'max_fps': 10,  # Limit processing FPS to reduce load
                'enable_display': True
            }
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Deep merge configurations
                self._deep_merge_dict(default_config, user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _deep_merge_dict(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value
    
    def initialize(self) -> bool:
        """Initialize all system components"""
        print("🚀 Initializing Integrated Object Detection + SLAM System")
        
        # Initialize object detector
        print("\n1. Loading object detection model...")
        model_loaded, model_msg = self.object_detector.load_model()
        if not model_loaded:
            print(f"❌ Failed to load object detection model: {model_msg}")
            return False
        print(f"✅ {model_msg}")
        
        # Initialize communication
        print("\n2. Initializing communication...")
        try:
            comm_config = self.config.get('communication', {})
            protocol = comm_config.get('protocol', 'socket')
            
            if protocol == 'socket':
                self.communication_handler = create_communication_handler(
                    protocol='socket',
                    host=comm_config.get('host', 'localhost'),
                    port=comm_config.get('port', 8888)
                )
            elif protocol == 'ros':
                self.communication_handler = create_communication_handler(
                    protocol='ros',
                    topic=comm_config.get('topic', '/detected_objects'),
                    node_name=comm_config.get('node_name', 'object_detector')
                )
            elif protocol == 'file':
                self.communication_handler = create_communication_handler(
                    protocol='file',
                    output_file=comm_config.get('output_file', 'detection_data.json')
                )
            else:
                print(f"❌ Unsupported communication protocol: {protocol}")
                return False
            
            print(f"✅ Communication initialized: {protocol}")
            
        except Exception as e:
            print(f"❌ Communication initialization failed: {e}")
            return False
        
        # Initialize camera
        print("\n3. Initializing camera...")
        if not self._initialize_camera():
            print("❌ Camera initialization failed")
            return False
        print("✅ Camera initialized")
        
        print("\n🎉 System initialization complete!")
        return True
    
    def _initialize_camera(self) -> bool:
        """Initialize camera (RealSense or webcam)"""
        camera_config = self.config.get('camera', {})
        
        if camera_config.get('use_realsense', True) and REALSENSE_AVAILABLE:
            try:
                self.camera_pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(
                    rs.stream.color,
                    camera_config.get('width', 640),
                    camera_config.get('height', 480),
                    rs.format.bgr8,
                    camera_config.get('fps', 30)
                )
                config.enable_stream(
                    rs.stream.depth,
                    camera_config.get('width', 640),
                    camera_config.get('height', 480),
                    rs.format.z16,
                    camera_config.get('fps', 30)
                )
                
                self.camera_pipeline.start(config)
                self.camera_type = 'realsense'
                print("Using Intel RealSense camera")
                return True
                
            except Exception as e:
                print(f"RealSense initialization failed: {e}")
                print("Falling back to webcam...")
        
        # Fallback to webcam
        try:
            self.camera_pipeline = cv2.VideoCapture(0)
            self.camera_pipeline.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('width', 640))
            self.camera_pipeline.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('height', 480))
            self.camera_pipeline.set(cv2.CAP_PROP_FPS, camera_config.get('fps', 30))
            self.camera_type = 'webcam'
            print("Using webcam (depth information not available)")
            return True
            
        except Exception as e:
            print(f"Webcam initialization failed: {e}")
            return False
    
    def get_camera_frames(self) -> tuple:
        """Get frames from camera"""
        if self.camera_type == 'realsense':
            try:
                frames = self.camera_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    return color_image, depth_image
                else:
                    return None, None
                    
            except Exception as e:
                print(f"RealSense frame capture failed: {e}")
                return None, None
        
        elif self.camera_type == 'webcam':
            try:
                ret, frame = self.camera_pipeline.read()
                if ret:
                    # Create dummy depth frame for webcam
                    dummy_depth = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
                    return frame, dummy_depth
                else:
                    return None, None
                    
            except Exception as e:
                print(f"Webcam frame capture failed: {e}")
                return None, None
        
        return None, None
    
    def process_frame(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> Dict:
        """Process a single frame through the entire pipeline"""
        with self.lock:
            # 1. Run object detection
            annotated_frame, detection_metadata = self.object_detector.process_frame(rgb_frame, depth_frame)
            
            if 'error' in detection_metadata:
                return {'error': detection_metadata['error'], 'frame': annotated_frame}
            
            # 2. Process detections for SLAM
            detections = detection_metadata.get('detections', [])
            slam_result = self.slam_manager.process_detection_data(detections, self.camera_pose)
            
            # 3. Send data to navigation system
            if self.communication_handler and detections:
                success = self.communication_handler.send_detection_data(detections)
                if success:
                    self.stats['detections_sent'] += 1
            
            # 4. Update statistics
            self.stats['frames_processed'] += 1
            self.stats['landmarks_created'] = slam_result['statistics']['landmarks_created']
            
            return {
                'frame': annotated_frame,
                'detections': detections,
                'landmarks': slam_result['landmarks'],
                'statistics': self.get_statistics()
            }
    
    def run_realtime(self):
        """Run real-time processing loop"""
        if not self.initialize():
            return
        
        print("\n🔄 Starting real-time processing...")
        print("Press 'q' to quit, 's' to save landmarks, 'r' to reset landmarks")
        
        self.running = True
        last_process_time = 0
        max_fps = self.config.get('processing', {}).get('max_fps', 10)
        min_frame_time = 1.0 / max_fps
        
        try:
            while self.running:
                current_time = time.time()
                
                # Limit processing FPS
                if current_time - last_process_time < min_frame_time:
                    time.sleep(0.01)
                    continue
                
                # Get camera frames
                rgb_frame, depth_frame = self.get_camera_frames()
                if rgb_frame is None:
                    continue
                
                # Process frame
                result = self.process_frame(rgb_frame, depth_frame)
                
                if 'error' in result:
                    print(f"Processing error: {result['error']}")
                    continue
                
                # Display results
                if self.config.get('processing', {}).get('enable_display', True):
                    self._display_results(result)
                
                # Print statistics periodically
                if self.stats['frames_processed'] % 30 == 0:
                    stats = result['statistics']
                    print(f"\n📊 Stats: {stats['frames_processed']} frames, "
                          f"{stats['detections_sent']} detections sent, "
                          f"{stats['landmarks_created']} landmarks created")
                
                last_process_time = current_time
                
                # Handle keyboard input
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif cv2.waitKey(1) & 0xFF == ord('s'):
                    self.slam_manager.slam_processor.export_landmarks("manual_save_landmarks.json")
                    print("💾 Landmarks saved manually")
                elif cv2.waitKey(1) & 0xFF == ord('r'):
                    self.slam_manager.slam_processor.reset_landmarks()
                    print("🔄 Landmarks reset")
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.shutdown()
    
    def _display_results(self, result: Dict):
        """Display processing results"""
        frame = result['frame']
        landmarks = result['landmarks']
        detections = result['detections']
        
        # Add landmark info to frame
        info_text = f"Landmarks: {len(landmarks)}, Detections: {len(detections)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Object Detection + SLAM', frame)
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        runtime = time.time() - self.stats['start_time']
        
        # Get individual component statistics
        detection_stats = self.object_detector.get_statistics()
        slam_stats = self.slam_manager.get_navigation_data()['statistics']
        
        comm_stats = {}
        if self.communication_handler:
            comm_stats = self.communication_handler.get_statistics()
        
        return {
            'system': {
                **self.stats,
                'runtime_seconds': runtime,
                'fps': self.stats['frames_processed'] / runtime if runtime > 0 else 0
            },
            'detection': detection_stats,
            'slam': slam_stats,
            'communication': comm_stats
        }
    
    def shutdown(self):
        """Shutdown all components"""
        print("\n🔄 Shutting down system...")
        self.running = False
        
        # Stop camera
        if self.camera_pipeline:
            if self.camera_type == 'realsense':
                self.camera_pipeline.stop()
            elif self.camera_type == 'webcam':
                self.camera_pipeline.release()
        
        # Close display
        cv2.destroyAllWindows()
        
        # Shutdown communication
        if self.communication_handler:
            self.communication_handler.shutdown()
        
        # Save final landmarks
        self.slam_manager.slam_processor.export_landmarks("final_landmarks.json")
        
        print("✅ System shutdown complete")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Integrated Object Detection + SLAM System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--protocol', type=str, default='socket', 
                       choices=['socket', 'ros', 'file'], help='Communication protocol')
    parser.add_argument('--host', type=str, default='localhost', help='Socket host')
    parser.add_argument('--port', type=int, default=8888, help='Socket port')
    parser.add_argument('--model', type=str, default='PredictModel/best PT 3500.pt', 
                       help='YOLO model path')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--max-fps', type=int, default=10, help='Maximum processing FPS')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = {
        'model_path': args.model,
        'communication': {
            'protocol': args.protocol,
            'host': args.host,
            'port': args.port
        },
        'processing': {
            'max_fps': args.max_fps,
            'enable_display': not args.no_display
        }
    }
    
    # Save temporary config file
    with open('temp_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create and run system
    system = IntegratedSystem(args.config or 'temp_config.json')
    
    try:
        system.run_realtime()
    except Exception as e:
        print(f"❌ System error: {e}")
        system.shutdown()

if __name__ == "__main__":
    main() 