# Object Detection + SLAM Integration Guide

This guide explains how to integrate object detection data with SLAM navigation systems for robotics applications.

## 🎯 Overview

This integration allows your object detection system to communicate with your teammate's SLAM navigation system by:

1. **Processing object detection data** from YOLO with depth information
2. **Converting to world coordinates** for SLAM use
3. **Communicating data** via multiple protocols (Socket, ROS, File)
4. **Creating landmarks** for navigation and obstacle avoidance

## 📁 Files Created

### Core Components

1. **`communication_handler.py`** - Handles data transmission to navigation system
2. **`slam_integration.py`** - Processes detection data for SLAM use
3. **`integration_example.py`** - Complete integrated system example
4. **`ros/scripts/navigation_receiver.py`** - Example receiver for navigation system

### Supporting Files

- **`object_detection_processor.py`** - Your existing detection processor (enhanced)
- **`README_Integration.md`** - This guide

## 🚀 Quick Start

### For Your Part (Object Detection)

1. **Run the integrated system:**
```bash
python integration_example.py --protocol socket --host localhost --port 8888
```

2. **Or use individual components:**
```python
from communication_handler import create_communication_handler
from slam_integration import SLAMIntegrationManager
from object_detection_processor import ObjectDetectionProcessor

# Initialize components
detector = ObjectDetectionProcessor()
slam_manager = SLAMIntegrationManager()
comm_handler = create_communication_handler("socket", host="localhost", port=8888)

# Process frame
annotated_frame, detection_data = detector.process_frame(rgb_frame, depth_frame)
slam_result = slam_manager.process_detection_data(detection_data['detections'])
comm_handler.send_detection_data(detection_data['detections'])
```

### For Your Teammate (Navigation System)

1. **Run the navigation receiver:**
```bash
python ros/scripts/navigation_receiver.py --host localhost --port 8888
```

2. **Or integrate into existing SLAM system:**
```python
from ros.scripts.navigation_receiver import NavigationReceiver

# Create receiver
receiver = NavigationReceiver("localhost", 8888)
receiver.start()

# Get landmarks for SLAM
landmarks = receiver.get_navigation_landmarks()
obstacle_map = receiver.get_obstacle_map()
```

## 🔧 Configuration

### Communication Protocols

Choose the best protocol for your setup:

#### 1. Socket Communication (Recommended for testing)
```python
comm_handler = create_communication_handler(
    protocol="socket",
    host="localhost",
    port=8888
)
```

#### 2. ROS Communication (For ROS-based systems)
```python
comm_handler = create_communication_handler(
    protocol="ros",
    topic="/detected_objects",
    node_name="object_detector"
)
```

#### 3. File Communication (For offline processing)
```python
comm_handler = create_communication_handler(
    protocol="file",
    output_file="detection_data.json"
)
```

### Camera Calibration

Update camera parameters in `slam_integration.py`:
```python
camera_calibration = CameraCalibration(
    fx=525.0,  # Focal length x
    fy=525.0,  # Focal length y
    cx=320.0,  # Principal point x
    cy=240.0,  # Principal point y
    width=640,
    height=480
)
```

## 📊 Data Format

### Object Detection Data Sent
```json
{
  "timestamp": 1234567890.123,
  "frame_id": "camera_link",
  "object_count": 2,
  "objects": [
    {
      "object_id": 1,
      "class_name": "chair",
      "confidence": 0.85,
      "timestamp": 1234567890.123,
      "bbox_2d": [100, 150, 200, 300],
      "image_center": [150, 225],
      "position_3d": [1.5, 0.2, 2.5],
      "depth_mm": 2500,
      "camera_frame": "camera_link",
      "is_static": true,
      "reliability_score": 0.9
    }
  ]
}
```

### SLAM Landmarks Received
```json
{
  "landmarks": [
    {
      "id": 1,
      "type": "chair",
      "position": [1.5, 0.2, 2.5],
      "confidence": 0.85,
      "uncertainty": 0.1,
      "observations": 5,
      "is_static": true
    }
  ],
  "timestamp": 1234567890.123
}
```

## 🔄 Integration Workflow

1. **Object Detection**: YOLO detects objects in RGB-D frames
2. **Depth Processing**: Calculate 3D position using depth data
3. **Coordinate Transformation**: Convert to world coordinates
4. **SLAM Processing**: Filter objects suitable for landmarks
5. **Communication**: Send data to navigation system
6. **Navigation Integration**: Use landmarks for SLAM and obstacle avoidance

## 🎛️ Command Line Usage

### Object Detection System
```bash
# Basic usage
python integration_example.py

# With custom settings
python integration_example.py \
    --protocol socket \
    --host 192.168.1.100 \
    --port 8888 \
    --model "path/to/your/model.pt" \
    --max-fps 15

# ROS mode
python integration_example.py \
    --protocol ros \
    --topic "/detected_objects" \
    --node-name "object_detector"

# File mode (offline)
python integration_example.py \
    --protocol file \
    --output-file "detection_data.json" \
    --no-display
```

### Navigation System
```bash
# Basic receiver
python ros/scripts/navigation_receiver.py

# Custom settings
python ros/scripts/navigation_receiver.py \
    --host 0.0.0.0 \
    --port 8888 \
    --export-interval 30
```

## 🛠️ Advanced Features

### Coordinate Transformation
```python
from slam_integration import CoordinateTransformer, CameraCalibration

# Initialize transformer
calibration = CameraCalibration(fx=525.0, fy=525.0, cx=320.0, cy=240.0)
transformer = CoordinateTransformer(calibration)

# Convert image point to 3D
image_point = (320, 240)  # Center of image
depth = 2.5  # meters
camera_coords = transformer.image_to_camera_coordinates(image_point, depth)

# Convert to world coordinates
camera_pose = [0, 0, 0, 0, 0, 0]  # [x, y, z, roll, pitch, yaw]
world_coords = transformer.camera_to_world_coordinates(camera_coords, camera_pose)
```

### Landmark Management
```python
from slam_integration import SLAMProcessor

# Initialize processor
slam_processor = SLAMProcessor()

# Process detections
detection_data = [...]  # Your detection data
result = slam_processor.process_detection_frame(detection_data, camera_pose)

# Get landmarks for navigation
nav_landmarks = slam_processor.get_landmarks_for_navigation()

# Export/import landmarks
slam_processor.export_landmarks("landmarks.json")
slam_processor.import_landmarks("previous_landmarks.json")
```

## 🔧 Troubleshooting

### Common Issues

1. **Socket Connection Failed**
   - Check if port is available: `netstat -an | grep 8888`
   - Try different port or host
   - Ensure firewall allows connection

2. **Model Loading Issues**
   - Verify model path exists
   - Check ultralytics version: `pip install ultralytics>=8.2.0`
   - Ensure PyTorch compatibility

3. **Camera Not Found**
   - Check RealSense connection: `python -c "import pyrealsense2 as rs; print('RealSense OK')"`
   - Verify webcam access: `python -c "import cv2; print(cv2.VideoCapture(0).read()[0])"`

4. **ROS Communication**
   - Ensure ROS is running: `roscore`
   - Check topic: `rostopic list`
   - Verify ROS Python packages installed

### Performance Optimization

1. **Reduce Processing Load**
   - Lower max FPS: `--max-fps 5`
   - Disable display: `--no-display`
   - Use smaller YOLO model

2. **Improve Accuracy**
   - Calibrate camera properly
   - Filter objects by confidence
   - Use multiple observations for landmarks

## 📝 Integration with Existing SLAM

### For Your Teammate's Navigation System

1. **Receive Object Data**:
```python
# Use the NavigationReceiver class
receiver = NavigationReceiver("localhost", 8888)
receiver.start()

# Get processed landmarks
landmarks = receiver.get_navigation_landmarks()
```

2. **Integrate with SLAM**:
```python
# Example integration with existing SLAM
for landmark in landmarks:
    slam_system.add_landmark(
        id=landmark['id'],
        position=landmark['position'],
        type=landmark['type'],
        confidence=landmark['confidence']
    )
```

3. **Use for Obstacle Avoidance**:
```python
# Get obstacle map
obstacle_map = receiver.get_obstacle_map()
path_planner.update_obstacles(obstacle_map['obstacles'])
```

## 🎯 Best Practices

1. **Data Quality**
   - Filter low-confidence detections
   - Use multiple observations for landmarks
   - Validate depth measurements

2. **Communication**
   - Handle connection failures gracefully
   - Buffer data during network issues
   - Use appropriate message rates

3. **Coordinate Systems**
   - Calibrate camera properly
   - Transform to common reference frame
   - Account for camera pose changes

4. **Performance**
   - Limit processing frequency
   - Use efficient data structures
   - Profile and optimize bottlenecks

## 🤝 Team Collaboration

### Division of Responsibilities

**Your Part (Object Detection)**:
- Run object detection on RGB-D frames
- Calculate object positions and depths
- Send processed data to navigation system
- Maintain object tracking and confidence scores

**Teammate's Part (Navigation/SLAM)**:
- Receive object detection data
- Integrate landmarks into SLAM map
- Use for path planning and obstacle avoidance
- Provide camera pose feedback (optional)

### Communication Protocol

Establish clear communication about:
- Data format and coordinate systems
- Communication protocol and connection details
- Error handling and recovery procedures
- Performance requirements and limitations

## 📚 Example Usage Scenarios

### Scenario 1: Real-time Navigation
```bash
# Terminal 1: Start navigation receiver
python ros/scripts/navigation_receiver.py --host localhost --port 8888

# Terminal 2: Start object detection
python integration_example.py --protocol socket --host localhost --port 8888
```

### Scenario 2: ROS Integration
```bash
# Terminal 1: Start ROS
roscore

# Terminal 2: Start object detection with ROS
python integration_example.py --protocol ros --topic /detected_objects

# Terminal 3: Your teammate's ROS navigation node
# rostopic echo /detected_objects
```

### Scenario 3: Offline Processing
```bash
# Generate detection data file
python integration_example.py --protocol file --output-file robot_session.json

# Process offline (your teammate)
python process_offline_data.py --input robot_session.json
```

This integration provides a robust foundation for combining object detection with SLAM navigation systems. The modular design allows for easy customization and extension based on your specific requirements.

## 🔗 Next Steps

1. Test the basic socket communication between systems
2. Calibrate your camera for accurate 3D positioning
3. Tune SLAM parameters for your specific environment
4. Implement error handling and recovery mechanisms
5. Optimize for your target performance requirements

Good luck with your group project! 🚀 