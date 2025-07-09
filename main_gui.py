import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QGridLayout, 
                            QGroupBox, QTextEdit, QProgressBar, QFileDialog,
                            QMessageBox, QFrame, QSizePolicy)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor
import cv2
import numpy as np
from datetime import datetime
import time

from config_loader import ConfigLoader
from camera_controller import CameraController
from dependency_manager import DependencyManager
from model_manager import ModelManager

class VideoDisplayWidget(QLabel):
    """Custom widget for displaying video frames"""
    
    def __init__(self, title: str, width: int = 640, height: int = 480):
        super().__init__()
        self.title = title
        self.display_width = width
        self.display_height = height
        
        # Setup widget
        self.setFixedSize(width, height)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                background-color: #f0f0f0;
                color: #666666;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"{title}\nNo Signal")
        
        # Create title label
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                border: none;
                background-color: rgba(0, 0, 0, 180);
                color: white;
                padding: 5px;
                font-weight: bold;
            }
        """)
        
    def update_frame(self, frame):
        """Update the displayed frame"""
        if frame is None:
            self.setText(f"{self.title}\nNo Signal")
            return
            
        # Convert frame to QImage
        if len(frame.shape) == 3:  # Color image
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:  # Grayscale
            height, width = frame.shape
            bytes_per_line = width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Scale to fit widget
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.display_width, self.display_height, 
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.setPixmap(scaled_pixmap)

class MainWindow(QMainWindow):
    """Main GUI application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intel RealSense D435i Live Viewer with Object Detection")
        self.setGeometry(100, 100, 1400, 800)
        
        # Load configuration
        self.config_loader = ConfigLoader()
        if not self.config_loader.is_valid():
            QMessageBox.critical(self, "Configuration Error", 
                               "Failed to load camera configuration from config.json")
        
        # Initialize camera controller
        self.camera_controller = CameraController(self.config_loader)
        
        # Initialize detection managers
        self.dependency_manager = DependencyManager()
        self.model_manager = ModelManager()
        
        # UI state
        self.is_camera_connected = False
        self.recording_active = False
        self.detection_enabled = True
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Setup UI
        self.setup_ui()
        self.setup_timer()
        
        # Print configuration summary
        self.config_loader.print_summary()
        
        # Setup object detection (after UI is ready)
        QTimer.singleShot(100, self.setup_object_detection)
    
    def setup_object_detection(self):
        """Setup object detection with dependency and model checking"""
        # Check dependencies first
        deps_available, missing = self.dependency_manager.check_detection_dependencies()
        
        if not deps_available:
            self.log_message(f"Object detection dependencies missing: {', '.join(missing)}")
            
            # Prompt user to install dependencies
            if self.dependency_manager.prompt_install_dependencies(self):
                self.log_message("Installing object detection dependencies...")
                success, message = self.dependency_manager.install_dependencies_with_progress(self)
                
                if success:
                    self.log_message("✓ Dependencies installed successfully. Restart required.")
                    QMessageBox.information(
                        self,
                        "Installation Complete",
                        "Dependencies installed successfully!\n\nPlease restart the application to use object detection."
                    )
                    return
                else:
                    self.log_message(f"❌ Failed to install dependencies: {message}")
                    return
            else:
                self.log_message("Object detection disabled - dependencies not installed")
                return
        
        # Dependencies are available, now check for model
        model_path = self.model_manager.prompt_model_selection(self)
        if not model_path:
            self.log_message("Object detection disabled - no model selected")
            return
        
        # Validate and load model
        is_valid, error_msg = self.model_manager.validate_model_file(model_path)
        if not is_valid:
            self.log_message(f"❌ Invalid model file: {error_msg}")
            QMessageBox.critical(self, "Model Error", f"Invalid model file:\n\n{error_msg}")
            return
        
        # Setup detection in camera controller
        success, message = self.camera_controller.setup_object_detection(model_path)
        if success:
            self.camera_controller.enable_detection(True)
            self.detection_enabled = True
            self.log_message(f"✓ Object detection enabled: {message}")
            
            # Update depth display to detection display
            self.depth_display.title = "Object Detection"
            self.depth_display.setText("Object Detection\nReady")
        else:
            self.log_message(f"❌ Failed to setup object detection: {message}")
            QMessageBox.critical(self, "Detection Error", f"Failed to setup object detection:\n\n{message}")
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video displays
        video_panel = self.create_video_panel()
        main_layout.addWidget(video_panel, stretch=3)
        
        # Right panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
        
        # Set dark theme
        self.apply_dark_theme()
    
    def create_video_panel(self):
        """Create the video display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Live Camera Feed")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Video displays container
        video_container = QWidget()
        video_layout = QHBoxLayout(video_container)
        
        # Get stream config for display size
        stream_config = self.config_loader.get_stream_config()
        display_width = min(stream_config['width'], 640)
        display_height = min(stream_config['height'], 480)
        
        # RGB display
        self.rgb_display = VideoDisplayWidget("RGB Camera", display_width, display_height)
        video_layout.addWidget(self.rgb_display)
        
        # Detection display (replaces depth display)
        self.depth_display = VideoDisplayWidget("Object Detection", display_width, display_height)
        video_layout.addWidget(self.depth_display)
        
        layout.addWidget(video_container)
        
        # Frame info display
        self.frame_info_label = QLabel("Frame: 0 | FPS: 0.0 | Status: Disconnected")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setStyleSheet("background-color: #333333; color: white; padding: 5px;")
        layout.addWidget(self.frame_info_label)
        
        # Detection stats display
        self.detection_stats_label = QLabel("Detection: Disabled | Objects: 0 | Processing: 0ms")
        self.detection_stats_label.setAlignment(Qt.AlignCenter)
        self.detection_stats_label.setStyleSheet("background-color: #2a4a2a; color: white; padding: 5px;")
        layout.addWidget(self.detection_stats_label)
        
        return panel
    
    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Camera controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout(camera_group)
        
        self.connect_btn = QPushButton("Connect Camera")
        self.connect_btn.clicked.connect(self.toggle_camera_connection)
        self.connect_btn.setMinimumHeight(40)
        camera_layout.addWidget(self.connect_btn)
        
        self.start_stream_btn = QPushButton("Start Streaming")
        self.start_stream_btn.clicked.connect(self.toggle_streaming)
        self.start_stream_btn.setEnabled(False)
        self.start_stream_btn.setMinimumHeight(40)
        camera_layout.addWidget(self.start_stream_btn)
        
        layout.addWidget(camera_group)
        
        # Recording controls
        recording_group = QGroupBox("Recording Controls")
        recording_layout = QVBoxLayout(recording_group)
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        self.record_btn.setMinimumHeight(40)
        recording_layout.addWidget(self.record_btn)
        
        self.save_path_btn = QPushButton("Select Save Location")
        self.save_path_btn.clicked.connect(self.select_save_location)
        recording_layout.addWidget(self.save_path_btn)
        
        self.save_path_label = QLabel("Save to: ./recordings/")
        self.save_path_label.setWordWrap(True)
        self.save_path_label.setStyleSheet("font-size: 10px; color: #666666;")
        recording_layout.addWidget(self.save_path_label)
        
        layout.addWidget(recording_group)
        
        # Camera info
        info_group = QGroupBox("Camera Information")
        info_layout = QVBoxLayout(info_group)
        
        device_info = self.config_loader.get_device_info()
        stream_config = self.config_loader.get_stream_config()
        
        info_text = f"""
Device: {device_info.get('name', 'Unknown')}
Firmware: {device_info.get('fw version', 'Unknown')}
Resolution: {stream_config['width']}x{stream_config['height']}
FPS: {stream_config['fps']}
Format: {stream_config['depth_format']}
        """.strip()
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        info_layout.addWidget(info_label)
        
        layout.addWidget(info_group)
        
        # Status log
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout(log_group)
        
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(150)
        self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet("font-family: monospace; font-size: 10px;")
        log_layout.addWidget(self.status_log)
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.status_log.clear)
        log_layout.addWidget(clear_log_btn)
        
        layout.addWidget(log_group)
        
        # Stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def setup_timer(self):
        """Setup timer for UI updates"""
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(33)  # ~30 FPS UI updates
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def log_message(self, message: str):
        """Add message to status log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.status_log.append(formatted_message)
        print(formatted_message)  # Also print to console
    
    def toggle_camera_connection(self):
        """Toggle camera connection"""
        if not self.is_camera_connected:
            self.log_message("Connecting to camera...")
            if self.camera_controller.connect():
                self.is_camera_connected = True
                self.connect_btn.setText("Disconnect Camera")
                self.start_stream_btn.setEnabled(True)
                self.log_message("✓ Camera connected successfully")
            else:
                self.log_message("❌ Failed to connect to camera")
                QMessageBox.critical(self, "Camera Error", 
                                   "Failed to connect to camera. Please check if the camera is plugged in and drivers are installed.")
        else:
            self.log_message("Disconnecting camera...")
            self.camera_controller.disconnect()
            self.is_camera_connected = False
            self.connect_btn.setText("Connect Camera")
            self.start_stream_btn.setText("Start Streaming")
            self.start_stream_btn.setEnabled(False)
            self.record_btn.setEnabled(False)
            
            # Clear displays
            self.rgb_display.update_frame(None)
            self.depth_display.update_frame(None)
            if hasattr(self, 'detection_stats_label'):
                self.detection_stats_label.setText("Detection: Disabled | Objects: 0 | Processing: 0ms")
                self.detection_stats_label.setStyleSheet("background-color: #4a2a2a; color: white; padding: 5px;")
            
            self.log_message("✓ Camera disconnected")
    
    def toggle_streaming(self):
        """Toggle video streaming"""
        if not self.camera_controller.is_streaming:
            self.log_message("Starting video stream...")
            if self.camera_controller.start_streaming():
                self.start_stream_btn.setText("Stop Streaming")
                self.record_btn.setEnabled(True)
                self.log_message("✓ Video streaming started")
            else:
                self.log_message("❌ Failed to start streaming")
        else:
            self.log_message("Stopping video stream...")
            self.camera_controller.stop_streaming()
            self.start_stream_btn.setText("Start Streaming")
            self.record_btn.setText("Start Recording")
            self.record_btn.setEnabled(False)
            self.recording_active = False
            
            # Clear displays
            self.rgb_display.update_frame(None)
            self.depth_display.update_frame(None)
            if hasattr(self, 'detection_stats_label'):
                self.detection_stats_label.setText("Detection: Disabled | Objects: 0 | Processing: 0ms")
                self.detection_stats_label.setStyleSheet("background-color: #4a2a2a; color: white; padding: 5px;")
            
            self.log_message("✓ Video streaming stopped")
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recording_active:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self.save_path_label.text().replace("Save to: ", "")
            
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            output_path = os.path.join(save_dir, f"recording_{timestamp}")
            
            self.log_message(f"Starting recording: {output_path}")
            if self.camera_controller.start_recording(output_path):
                self.recording_active = True
                self.record_btn.setText("Stop Recording")
                self.record_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #cc4444;
                        border: 1px solid #aa3333;
                    }
                    QPushButton:hover {
                        background-color: #dd5555;
                    }
                """)
                self.log_message("✓ Recording started")
            else:
                self.log_message("❌ Failed to start recording")
        else:
            self.log_message("Stopping recording...")
            self.camera_controller.stop_recording()
            self.recording_active = False
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("")  # Reset to default style
            self.log_message("✓ Recording stopped")
    
    def select_save_location(self):
        """Select save location for recordings"""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Location", "./recordings/")
        if folder:
            self.save_path_label.setText(f"Save to: {folder}/")
    
    def update_ui(self):
        """Update UI with latest frames and status"""
        if not self.camera_controller.is_streaming:
            return
        
        # Get latest frames
        frames = self.camera_controller.get_latest_frames()
        if frames:
            # Update displays
            self.rgb_display.update_frame(frames['rgb'])
            
            # Show detection or depth based on detection status
            if self.detection_enabled and 'annotated' in frames:
                self.depth_display.update_frame(frames['annotated'])
            else:
                self.depth_display.update_frame(frames['depth_colorized'])
            
            # Update frame counter
            self.frame_count += 1
            self.fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                fps = self.fps_counter / (current_time - self.last_fps_time)
                self.fps_counter = 0
                self.last_fps_time = current_time
                
                # Update status
                recording_type = "Detection Recording" if (self.recording_active and self.detection_enabled) else "Recording" if self.recording_active else "Streaming"
                self.frame_info_label.setText(f"Frame: {self.frame_count} | FPS: {fps:.1f} | Status: {recording_type}")
                
                # Update detection statistics
                if self.detection_enabled and self.camera_controller.is_detection_enabled():
                    stats = self.camera_controller.get_detection_stats()
                    detection_status = "Enabled"
                    objects_count = stats.get('last_detection_count', 0)
                    processing_time = stats.get('avg_processing_time', 0)
                    self.detection_stats_label.setText(f"Detection: {detection_status} | Objects: {objects_count} | Processing: {processing_time:.1f}ms")
                    self.detection_stats_label.setStyleSheet("background-color: #2a4a2a; color: white; padding: 5px;")  # Green background
                else:
                    self.detection_stats_label.setText("Detection: Disabled | Objects: 0 | Processing: 0ms")
                    self.detection_stats_label.setStyleSheet("background-color: #4a2a2a; color: white; padding: 5px;")  # Red background
    
    def closeEvent(self, event):
        """Handle application close"""
        self.log_message("Shutting down application...")
        if self.camera_controller:
            self.camera_controller.disconnect()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("RealSense D435i Viewer")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 