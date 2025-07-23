"""
Live 3D Position Visualizer for Object Detection
Displays real-time 3D position data for detected objects
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QTableWidget, QTableWidgetItem, QGroupBox,
                            QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from typing import Dict, List, Optional
import numpy as np

class Live3DVisualizer(QWidget):
    """Widget for displaying live 3D position data of detected objects"""
    
    def __init__(self):
        super().__init__()
        self.detection_data = []
        self.setup_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # Update every 100ms
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Live 3D Position Data")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #ffffff; padding: 10px; background-color: #404040; border-radius: 5px;")
        layout.addWidget(title)
        
        # Detection count
        self.count_label = QLabel("Objects Detected: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("color: #ffffff; padding: 5px; background-color: #2a4a2a; border-radius: 3px;")
        layout.addWidget(self.count_label)
        
        # Table for 3D data
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(8)
        self.data_table.setHorizontalHeaderLabels([
            "ID", "Object", "X (m)", "Y (m)", "Z (m)", "Azimuth (°)", "Elevation (°)", "Confidence"
        ])
        
        # Style the table
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                gridline-color: #404040;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #404040;
            }
            QTableWidget::item:selected {
                background-color: #505050;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # Make table headers resize to content
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.data_table)
        
        # Summary statistics
        stats_group = QGroupBox("Position Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No objects detected")
        self.stats_label.setFont(QFont("Consolas", 9))
        self.stats_label.setStyleSheet("color: #ffffff; background-color: #1e1e1e; padding: 10px; border-radius: 3px;")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        # Apply dark theme
        self.setStyleSheet("""
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
        """)
    
    def update_detection_data(self, frame_data: Dict):
        """Update with latest detection data"""
        if not frame_data or 'detection_metadata' not in frame_data:
            self.detection_data = []
            return
        
        detection_metadata = frame_data['detection_metadata']
        if not detection_metadata or 'detections' not in detection_metadata:
            self.detection_data = []
            return
        
        self.detection_data = detection_metadata['detections']
    
    def update_display(self):
        """Update the display with current detection data"""
        # Update object count
        count = len(self.detection_data)
        self.count_label.setText(f"Objects Detected: {count}")
        
        if count == 0:
            self.data_table.setRowCount(0)
            self.stats_label.setText("No objects detected")
            return
        
        # Update table
        self.data_table.setRowCount(count)
        
        valid_positions = []
        for i, detection in enumerate(self.detection_data):
            # Get 3D position data
            pos_3d = detection.get('position_3d', {})
            
            # Populate table row
            self.data_table.setItem(i, 0, QTableWidgetItem(str(detection.get('tracker_id', 'N/A'))))
            self.data_table.setItem(i, 1, QTableWidgetItem(detection.get('class_name', 'Unknown')))
            
            if pos_3d.get('valid', False):
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{pos_3d['x']:.3f}"))
                self.data_table.setItem(i, 3, QTableWidgetItem(f"{pos_3d['y']:.3f}"))
                self.data_table.setItem(i, 4, QTableWidgetItem(f"{pos_3d['z']:.3f}"))
                self.data_table.setItem(i, 5, QTableWidgetItem(f"{pos_3d['azimuth']:.1f}"))
                self.data_table.setItem(i, 6, QTableWidgetItem(f"{pos_3d['elevation']:.1f}"))
                
                # Color code based on distance
                distance = pos_3d['z']
                if distance < 1.0:
                    color = QColor(255, 100, 100)  # Red for close
                elif distance < 3.0:
                    color = QColor(255, 255, 100)  # Yellow for medium
                else:
                    color = QColor(100, 255, 100)  # Green for far
                
                for col in range(2, 7):  # Color distance-related columns
                    item = self.data_table.item(i, col)
                    if item:
                        item.setForeground(color)
                
                valid_positions.append(pos_3d)
            else:
                for col in range(2, 7):
                    self.data_table.setItem(i, col, QTableWidgetItem("N/A"))
                    item = self.data_table.item(i, col)
                    if item:
                        item.setForeground(QColor(128, 128, 128))  # Gray for invalid
            
            # Confidence
            confidence = detection.get('confidence', 0)
            conf_item = QTableWidgetItem(f"{confidence:.2f}" if confidence else "N/A")
            
            # Color code confidence
            if confidence and confidence > 0.8:
                conf_item.setForeground(QColor(100, 255, 100))  # Green for high confidence
            elif confidence and confidence > 0.5:
                conf_item.setForeground(QColor(255, 255, 100))  # Yellow for medium
            else:
                conf_item.setForeground(QColor(255, 100, 100))  # Red for low
            
            self.data_table.setItem(i, 7, conf_item)
        
        # Update statistics
        self._update_statistics(valid_positions)
        
        # Resize columns to content
        self.data_table.resizeColumnsToContents()
    
    def _update_statistics(self, valid_positions: List[Dict]):
        """Update position statistics"""
        if not valid_positions:
            self.stats_label.setText("No valid position data")
            return
        
        # Calculate statistics
        distances = [pos['z'] for pos in valid_positions]
        azimuths = [pos['azimuth'] for pos in valid_positions]
        elevations = [pos['elevation'] for pos in valid_positions]
        
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        avg_azimuth = np.mean(azimuths)
        avg_elevation = np.mean(elevations)
        
        # Find closest and farthest objects
        closest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)
        
        stats_text = f"""Position Statistics ({len(valid_positions)} valid objects):

Distance Range: {min_distance:.2f}m - {max_distance:.2f}m
Average Distance: {avg_distance:.2f}m

Average Azimuth: {avg_azimuth:.1f}°
Average Elevation: {avg_elevation:.1f}°

Closest Object: {min_distance:.2f}m (Az: {azimuths[closest_idx]:.1f}°)
Farthest Object: {max_distance:.2f}m (Az: {azimuths[farthest_idx]:.1f}°)

Spatial Distribution:
- Left Side: {sum(1 for az in azimuths if az < -10)} objects
- Center: {sum(1 for az in azimuths if -10 <= az <= 10)} objects  
- Right Side: {sum(1 for az in azimuths if az > 10)} objects
"""
        
        self.stats_label.setText(stats_text)


class CompactPositionDisplay(QWidget):
    """Compact widget for showing 3D position info in main window"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup compact display"""
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("3D Position Info")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 10, QFont.Bold))
        title.setStyleSheet("color: #ffffff; background-color: #404040; padding: 3px; border-radius: 3px;")
        layout.addWidget(title)
        
        # Scrollable area for object list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(2)
        
        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll)
        
        # Status
        self.status_label = QLabel("No objects detected")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #cccccc; font-size: 10px; padding: 3px;")
        layout.addWidget(self.status_label)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QScrollArea {
                border: 1px solid #404040;
                background-color: #1e1e1e;
            }
        """)
    
    def update_detection_data(self, frame_data: Dict):
        """Update with latest detection data"""
        # Clear existing widgets
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        
        if not frame_data or 'detection_metadata' not in frame_data:
            self.status_label.setText("No detection data")
            return
        
        detection_metadata = frame_data['detection_metadata']
        if not detection_metadata or 'detections' not in detection_metadata:
            self.status_label.setText("No objects detected")
            return
        
        detections = detection_metadata['detections']
        valid_count = 0
        
        for detection in detections:
            pos_3d = detection.get('position_3d', {})
            
            if pos_3d.get('valid', False):
                # Create compact info widget
                info_widget = QFrame()
                info_widget.setFrameStyle(QFrame.Box)
                info_widget.setStyleSheet("QFrame { border: 1px solid #404040; border-radius: 3px; padding: 2px; margin: 1px; }")
                
                info_layout = QVBoxLayout(info_widget)
                info_layout.setSpacing(1)
                info_layout.setContentsMargins(5, 2, 5, 2)
                
                # Object info
                obj_label = QLabel(f"#{detection.get('tracker_id', '?')} {detection.get('class_name', 'Unknown')}")
                obj_label.setFont(QFont("Arial", 9, QFont.Bold))
                obj_label.setStyleSheet("color: #ffffff;")
                info_layout.addWidget(obj_label)
                
                # Position info
                pos_label = QLabel(f"XYZ: ({pos_3d['x']:.2f}, {pos_3d['y']:.2f}, {pos_3d['z']:.2f})m")
                pos_label.setFont(QFont("Consolas", 8))
                pos_label.setStyleSheet("color: #cccccc;")
                info_layout.addWidget(pos_label)
                
                # Angles
                angle_label = QLabel(f"Az: {pos_3d['azimuth']:.1f}°  El: {pos_3d['elevation']:.1f}°")
                angle_label.setFont(QFont("Consolas", 8))
                angle_label.setStyleSheet("color: #cccccc;")
                info_layout.addWidget(angle_label)
                
                self.content_layout.addWidget(info_widget)
                valid_count += 1
        
        self.status_label.setText(f"{valid_count}/{len(detections)} objects with 3D data")
        
        # Add stretch to push items to top
        self.content_layout.addStretch()
