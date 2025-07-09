import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
from queue import Queue, Empty
from typing import Optional, Tuple, Callable
from config_loader import ConfigLoader
from object_detection_processor import ObjectDetectionProcessor

class CameraController:
    """Manages Intel RealSense D435i camera operations"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.pipeline = None
        self.config = None
        self.align = None
        
        # Streaming state
        self.is_streaming = False
        self.stream_thread = None
        self.frame_queue = Queue(maxsize=5)  # Limit queue size to prevent memory buildup
        
        # Callbacks for frame updates
        self.rgb_callback: Optional[Callable] = None
        self.depth_callback: Optional[Callable] = None
        
        # Frame recording
        self.recording = False
        self.rgb_writer = None
        self.depth_writer = None
        self.record_start_time = None
        
        # Object detection
        self.detection_processor = ObjectDetectionProcessor()
        self.detection_enabled = False
        self.detection_writer = None  # For recording annotated frames
        
        # Stream configuration
        self.stream_config = config_loader.get_stream_config()
        self.camera_params = config_loader.get_camera_parameters()
        
    def set_callbacks(self, rgb_callback: Callable, depth_callback: Callable):
        """Set callbacks for frame updates"""
        self.rgb_callback = rgb_callback
        self.depth_callback = depth_callback
    
    def setup_object_detection(self, model_path: str) -> Tuple[bool, str]:
        """Setup object detection with specified model path"""
        return self.detection_processor.load_model(model_path)
    
    def enable_detection(self, enabled: bool = True):
        """Enable or disable object detection processing"""
        self.detection_enabled = enabled and self.detection_processor.detection_stats['model_loaded']
    
    def is_detection_enabled(self) -> bool:
        """Check if object detection is enabled and working"""
        return self.detection_enabled and self.detection_processor.detection_stats['model_loaded']
    
    def get_detection_stats(self) -> dict:
        """Get current detection statistics"""
        return self.detection_processor.get_statistics()
    
    def connect(self) -> bool:
        """Connect to RealSense camera and start streaming"""
        try:
            # Initialize pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.depth, 
                                    self.stream_config['width'], 
                                    self.stream_config['height'], 
                                    rs.format.z16, 
                                    self.stream_config['fps'])
            
            self.config.enable_stream(rs.stream.color, 
                                    self.stream_config['width'], 
                                    self.stream_config['height'], 
                                    rs.format.bgr8, 
                                    self.stream_config['fps'])
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Create alignment object (align depth to color)
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # Apply camera parameters
            self._apply_camera_settings(profile)
            
            print("✓ RealSense camera connected successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to camera: {e}")
            self.disconnect()
            return False
    
    def _apply_camera_settings(self, profile):
        """Apply camera settings from configuration"""
        try:
            # Get device sensors
            device = profile.get_device()
            
            # Color sensor settings
            if device.query_sensors():
                color_sensor = None
                depth_sensor = None
                
                for sensor in device.query_sensors():
                    if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                        color_sensor = sensor
                    elif sensor.get_info(rs.camera_info.name) == 'Stereo Module':
                        depth_sensor = sensor
                
                # Apply color settings
                if color_sensor and color_sensor.supports(rs.option.exposure):
                    if not self.camera_params.get("color_auto_exposure", True):
                        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                        if "color_exposure" in self.camera_params:
                            color_sensor.set_option(rs.option.exposure, self.camera_params["color_exposure"])
                    else:
                        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                
                if color_sensor and color_sensor.supports(rs.option.gain):
                    if "color_gain" in self.camera_params:
                        color_sensor.set_option(rs.option.gain, self.camera_params["color_gain"])
                
                if color_sensor and color_sensor.supports(rs.option.enable_auto_white_balance):
                    auto_wb = self.camera_params.get("auto_white_balance", True)
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 1 if auto_wb else 0)
                    if not auto_wb and "white_balance" in self.camera_params:
                        color_sensor.set_option(rs.option.white_balance, self.camera_params["white_balance"])
                
                # Apply depth settings
                if depth_sensor and depth_sensor.supports(rs.option.gain):
                    if "depth_gain" in self.camera_params:
                        depth_sensor.set_option(rs.option.gain, self.camera_params["depth_gain"])
                
                if depth_sensor and depth_sensor.supports(rs.option.laser_power):
                    if "laser_power" in self.camera_params:
                        depth_sensor.set_option(rs.option.laser_power, self.camera_params["laser_power"])
                
                if depth_sensor and depth_sensor.supports(rs.option.emitter_enabled):
                    laser_enabled = self.camera_params.get("laser_enabled", True)
                    depth_sensor.set_option(rs.option.emitter_enabled, 1 if laser_enabled else 0)
            
            print("✓ Camera settings applied from configuration")
            
        except Exception as e:
            print(f"⚠ Warning: Could not apply some camera settings: {e}")
    
    def start_streaming(self) -> bool:
        """Start camera streaming in separate thread"""
        if self.is_streaming or not self.pipeline:
            return False
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        print("✓ Camera streaming started")
        return True
    
    def stop_streaming(self):
        """Stop camera streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        # Stop recording if active
        self.stop_recording()
        
        print("✓ Camera streaming stopped")
    
    def _stream_worker(self):
        """Worker thread for camera streaming"""
        while self.is_streaming and self.pipeline:
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # Align depth to color
                aligned_frames = self.align.process(frames)
                
                # Get aligned frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create colorized depth image
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Process with object detection if enabled
                annotated_frame = color_image
                detection_metadata = None
                
                if self.detection_enabled:
                    try:
                        # Convert BGR to RGB for YOLO processing
                        rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                        annotated_frame, detection_metadata = self.detection_processor.process_frame(rgb_frame, depth_image)
                        # Convert back to BGR for display
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"⚠ Detection processing error: {e}")
                        annotated_frame = color_image
                
                # Put frames in queue (non-blocking)
                try:
                    frame_data = {
                        'rgb': color_image,
                        'depth': depth_image,
                        'depth_colorized': depth_colormap,
                        'annotated': annotated_frame,
                        'detection_metadata': detection_metadata,
                        'timestamp': time.time()
                    }
                    self.frame_queue.put_nowait(frame_data)
                except:
                    # Queue is full, skip this frame
                    pass
                
                # Handle recording
                if self.recording:
                    # Record annotated frame if detection is enabled, otherwise record RGB
                    frame_to_record = annotated_frame if self.detection_enabled else color_image
                    self._record_frame(frame_to_record, depth_colormap)
                    
                    # Store detection metadata for later saving
                    if self.detection_enabled and detection_metadata:
                        self.detection_processor.add_detection_metadata(detection_metadata)
                
                # Call callbacks if available
                if self.rgb_callback:
                    self.rgb_callback(color_image)
                if self.depth_callback:
                    self.depth_callback(depth_colormap)
                    
            except Exception as e:
                if self.is_streaming:  # Only print error if we're supposed to be streaming
                    print(f"⚠ Streaming error: {e}")
                break
    
    def get_latest_frames(self) -> Optional[dict]:
        """Get latest frame from queue (non-blocking)"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def start_recording(self, output_path: str = "recording") -> bool:
        """Start recording RGB/detection and depth streams"""
        if self.recording:
            return False
        
        try:
            # Setup video writers
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = self.stream_config['fps']
            size = (self.stream_config['width'], self.stream_config['height'])
            
            # Main recording writer (RGB or annotated detection frames)
            video_suffix = "_detection.avi" if self.detection_enabled else "_rgb.avi"
            self.rgb_writer = cv2.VideoWriter(
                f"{output_path}{video_suffix}", fourcc, fps, size
            )
            
            # Depth writer (optional, for backward compatibility)
            self.depth_writer = cv2.VideoWriter(
                f"{output_path}_depth.avi", fourcc, fps, size
            )
            
            # Clear detection metadata for new recording
            if self.detection_enabled:
                self.detection_processor.clear_metadata()
            
            self.recording = True
            self.record_start_time = time.time()
            self.recording_output_path = output_path
            
            recording_type = "detection" if self.detection_enabled else "RGB"
            print(f"✓ {recording_type} recording started: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording"""
        if not self.recording:
            return
        
        self.recording = False
        
        if self.rgb_writer:
            self.rgb_writer.release()
            self.rgb_writer = None
            
        if self.depth_writer:
            self.depth_writer.release()
            self.depth_writer = None
        
        # Save detection metadata if detection was enabled
        if self.detection_enabled and hasattr(self, 'recording_output_path'):
            video_suffix = "_detection.avi"
            output_file = f"{self.recording_output_path}{video_suffix}"
            success = self.detection_processor.save_detection_metadata(output_file)
            if success:
                print("✓ Detection metadata saved")
        
        if self.record_start_time:
            duration = time.time() - self.record_start_time
            recording_type = "Detection" if self.detection_enabled else "RGB"
            print(f"✓ {recording_type} recording stopped. Duration: {duration:.1f}s")
            self.record_start_time = None
    
    def _record_frame(self, rgb_frame, depth_frame):
        """Record current frame to video files"""
        try:
            if self.rgb_writer:
                self.rgb_writer.write(rgb_frame)
            if self.depth_writer:
                self.depth_writer.write(depth_frame)
        except Exception as e:
            print(f"⚠ Recording error: {e}")
    
    def disconnect(self):
        """Disconnect from camera and cleanup"""
        self.stop_streaming()
        
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None
        
        self.config = None
        self.align = None
        
        print("✓ Camera disconnected")
    
    def is_connected(self) -> bool:
        """Check if camera is connected"""
        return self.pipeline is not None
    
    def get_frame_info(self) -> dict:
        """Get current streaming information"""
        return {
            "connected": self.is_connected(),
            "streaming": self.is_streaming,
            "recording": self.recording,
            "fps": self.stream_config['fps'],
            "resolution": f"{self.stream_config['width']}x{self.stream_config['height']}",
            "queue_size": self.frame_queue.qsize()
        } 