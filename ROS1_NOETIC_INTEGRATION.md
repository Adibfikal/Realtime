# ROS1 Noetic Integration Guide

This guide explains how to integrate the realtime object detection system with ROS1 Noetic for navigation and SLAM systems.

## Overview

The system has been adapted to work with ROS1 Noetic, providing:
- **Custom ROS messages** for structured object detection data
- **Real-time communication** between detection and navigation systems
- **Visualization support** with RViz
- **Multiple communication protocols** (ROS1 Noetic, Socket, File)
- **Navigation-ready data** including landmarks and obstacles

## Prerequisites

### 1. ROS1 Noetic Installation

First, install ROS1 Noetic on your system:

```bash
# Ubuntu 20.04 LTS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-noetic-desktop-full

# Source ROS environment
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install additional dependencies
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo rosdep init
rosdep update
```

### 2. Workspace Setup

Create a catkin workspace for the detection system:

```bash
mkdir -p ~/detection_ws/src
cd ~/detection_ws/src

# Clone or copy the detection system
# (Assuming your detection system is in ~/detection_ws/src/realtime_detection)

# Install Python dependencies
pip3 install -r requirements.txt
```

### 3. Build Custom Messages

```bash
cd ~/detection_ws/src/realtime_detection
mkdir -p msg

# Copy message files
cp ros/msgs/DetectedObject.msg msg/
cp ros/msgs/DetectedObjects.msg msg/
cp ros/msgs/CMakeLists.txt ./
cp ros/msgs/package.xml ./

# Build the workspace
cd ~/detection_ws
catkin_make

# Source the workspace
source devel/setup.bash
echo "source ~/detection_ws/devel/setup.bash" >> ~/.bashrc
```

## System Architecture

### Components

1. **Detection System (Publisher)**
   - Runs the camera and object detection
   - Publishes detection data to ROS topics
   - Provides visualization markers

2. **Navigation Receiver (Subscriber)**
   - Subscribes to detection topics
   - Processes data for navigation/SLAM
   - Publishes landmarks and obstacles

3. **Visualization (RViz)**
   - Displays detection markers
   - Shows navigation landmarks
   - Visualizes obstacles and occupancy grid

### ROS Topics

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/detected_objects` | `DetectedObjects` | Raw detection data |
| `/detection_markers` | `MarkerArray` | Visualization markers |
| `/detection_pointcloud` | `PointCloud2` | Point cloud of detections |
| `/navigation_landmarks` | `MarkerArray` | Processed landmarks |
| `/navigation_obstacles` | `MarkerArray` | Obstacles for planning |
| `/navigation_occupancy_grid` | `OccupancyGrid` | Occupancy grid map |

### TF Frames

- `map` - World/global frame
- `camera_link` - Camera frame

## Usage

### Quick Start

1. **Terminal 1: Start ROS Core**
   ```bash
   roscore
   ```

2. **Terminal 2: Start Detection System**
   ```bash
   cd ~/detection_ws
   source devel/setup.bash
   python3 main.py
   ```
   - Enable object detection in GUI
   - Select "ros_noetic" protocol
   - Check "Enable Communication"

3. **Terminal 3: Start Navigation Receiver**
   ```bash
   cd ~/detection_ws
   source devel/setup.bash
   python3 ros/scripts/ros_noetic_receiver.py
   ```

4. **Terminal 4: Start RViz**
   ```bash
   rosrun rviz rviz -d ros/config/detection_visualization.rviz
   ```

### Using Launch Files

The system includes several launch files for different use cases:

#### Complete System
```bash
roslaunch realtime_detection realtime_detection.launch
```

#### Detection Sender Only
```bash
roslaunch realtime_detection detection_sender.launch
# Then manually start: python3 main.py
```

#### Navigation Receiver Only
```bash
roslaunch realtime_detection navigation_receiver.launch
```

### Launch File Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_frame` | `camera_link` | Camera TF frame |
| `world_frame` | `map` | World TF frame |
| `objects_topic` | `/detected_objects` | Detection data topic |
| `markers_topic` | `/detection_markers` | Visualization markers |
| `pointcloud_topic` | `/detection_pointcloud` | Point cloud topic |
| `landmarks_topic` | `/navigation_landmarks` | Landmarks topic |
| `obstacles_topic` | `/navigation_obstacles` | Obstacles topic |
| `use_rviz` | `true` | Start RViz visualization |
| `use_custom_msgs` | `true` | Use custom message types |

## GUI Integration

The main GUI application (`main.py`) now includes ROS communication controls:

### Communication Controls
- **Protocol Selection**: Choose between socket, ros, ros_noetic, file
- **Enable Communication**: Checkbox to enable/disable ROS communication
- **Status Display**: Shows communication status and statistics

### Setup in GUI
1. Start the application: `python3 main.py`
2. Connect camera and enable object detection
3. In "ROS Communication" section:
   - Select "ros_noetic" protocol
   - Check "Enable Communication"
4. Status should show "Active (ros_noetic)"

## API Reference

### ROS1NoeticCommunication Class

```python
from ros.scripts.ros_noetic_communication import ROS1NoeticCommunication

# Initialize
comm = ROS1NoeticCommunication(
    node_name="realtime_object_detector",
    objects_topic="/detected_objects",
    markers_topic="/detection_markers",
    pointcloud_topic="/detection_pointcloud",
    tf_frame="camera_link",
    world_frame="map"
)

# Connect
comm.connect()

# Send detection data
detection_results = [...]  # List of detection dictionaries
comm.send_detection_data(detection_results)

# Get statistics
stats = comm.get_statistics()
```

### ROS1NoeticReceiver Class

```python
from ros.scripts.ros_noetic_receiver import ROS1NoeticReceiver

# Initialize
receiver = ROS1NoeticReceiver(
    node_name="navigation_receiver",
    objects_topic="/detected_objects",
    landmarks_topic="/navigation_landmarks",
    obstacles_topic="/navigation_obstacles"
)

# Start
receiver.start()

# Get processed data
landmarks = receiver.get_landmarks_for_slam()
obstacles = receiver.get_obstacles_for_planning()
```

## Message Formats

### DetectedObject.msg
```
# Object identification
int32 object_id
string class_name
float32 confidence
time timestamp

# 2D bounding box in image coordinates
float32[4] bbox_2d
float32[2] image_center

# 3D position in camera frame
geometry_msgs/Point position_3d
float32 depth_mm

# Camera information
string camera_frame
int32 image_width
int32 image_height

# SLAM-specific information
bool is_static
float32 reliability_score
float32[3] position_uncertainty
```

### DetectedObjects.msg
```
# Standard ROS header
std_msgs/Header header

# Frame information
string frame_id
int32 object_count

# Array of detected objects
DetectedObject[] objects

# Detection statistics
float32 processing_time_ms
float32 detection_fps
bool detection_enabled
```

## Integration with Navigation Systems

### For SLAM Systems
```python
# Get landmarks
landmarks = receiver.get_landmarks_for_slam()

for landmark in landmarks:
    # Use landmark data for SLAM
    position = landmark['position']  # [x, y, z]
    confidence = landmark['confidence']
    uncertainty = landmark['uncertainty']
    object_type = landmark['type']
```

### For Path Planning
```python
# Get obstacles
obstacles = receiver.get_obstacles_for_planning()

for obstacle in obstacles:
    # Use obstacle data for path planning
    position = obstacle['position']  # [x, y, z]
    radius = obstacle['radius']
    confidence = obstacle['confidence']
```

### Occupancy Grid
The system publishes an occupancy grid on `/navigation_occupancy_grid` that can be used directly with navigation stacks like `move_base`.

## Troubleshooting

### Common Issues

1. **Custom Messages Not Found**
   ```bash
   # Rebuild workspace
   cd ~/detection_ws
   catkin_make
   source devel/setup.bash
   ```

2. **Communication Fails**
   - Check ROS is running: `roscore`
   - Verify topics: `rostopic list`
   - Check node status: `rosnode list`

3. **No Detection Data**
   - Ensure object detection is enabled in GUI
   - Check camera connection
   - Verify model is loaded

4. **TF Errors**
   - Check static transform publisher is running
   - Verify frame names match configuration

### Debug Commands

```bash
# Check ROS topics
rostopic list
rostopic info /detected_objects
rostopic echo /detected_objects

# Check TF tree
rosrun tf view_frames
rosrun tf tf_echo map camera_link

# Monitor node status
rosnode list
rosnode info /realtime_object_detector
```

## Performance Considerations

### Optimization Tips

1. **Reduce Message Frequency**
   - Filter low-confidence detections
   - Limit publishing rate

2. **Optimize Processing**
   - Use efficient detection models
   - Reduce image resolution if needed

3. **Network Considerations**
   - Use compression for large messages
   - Consider message priorities

### Monitoring Performance

```python
# Check communication statistics
stats = comm.get_statistics()
print(f"Messages sent: {stats['messages_sent']}")
print(f"Messages/sec: {stats['messages_per_second']}")
print(f"Objects/sec: {stats['objects_per_second']}")
```

## Example Integration

Here's a complete example of integrating with your navigation system:

```python
#!/usr/bin/env python3
import rospy
from ros.scripts.ros_noetic_receiver import ROS1NoeticReceiver

class NavigationSystem:
    def __init__(self):
        self.receiver = ROS1NoeticReceiver()
        self.receiver.start()
        
        # Your navigation system initialization
        self.slam_system = YourSLAMSystem()
        self.path_planner = YourPathPlanner()
        
        # Timer for processing
        rospy.Timer(rospy.Duration(0.1), self.process_detection_data)
    
    def process_detection_data(self, event):
        # Get landmarks for SLAM
        landmarks = self.receiver.get_landmarks_for_slam()
        for landmark in landmarks:
            self.slam_system.add_landmark(landmark)
        
        # Get obstacles for path planning
        obstacles = self.receiver.get_obstacles_for_planning()
        self.path_planner.update_obstacles(obstacles)

if __name__ == "__main__":
    rospy.init_node('navigation_system')
    nav_system = NavigationSystem()
    rospy.spin()
```

## Future Enhancements

Potential improvements for the ROS1 Noetic integration:

1. **Advanced TF Integration**
   - Dynamic camera pose estimation
   - Multiple camera support

2. **Enhanced Message Types**
   - Semantic segmentation data
   - Object tracking trajectories

3. **Performance Optimization**
   - Message compression
   - Adaptive publishing rates

4. **Navigation Features**
   - Dynamic obstacle avoidance
   - Semantic mapping integration

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review ROS logs: `rosrun rqt_console rqt_console`
3. Verify system requirements
4. Test with provided examples

## License

This ROS1 Noetic integration follows the same license as the main project. 