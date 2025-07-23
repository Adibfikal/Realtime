"""
README: 3D Point Cloud Positioning Implementation

This implementation adds x, y, z coordinates, azimuth, and elevation angle calculation 
for detected objects in the live RealSense camera feed.

## New Features Added:

### 1. Enhanced Object Detection Processor
- `calculate_3d_position_and_angles()`: Calculates 3D position and angles for detected objects
- Enhanced `process_frame()`: Now includes 3D positioning data in detection metadata
- Uses camera intrinsic parameters for accurate 3D calculations

### 2. Live 3D Visualization Components
- `Live3DVisualizer`: Full table view with statistics
- `CompactPositionDisplay`: Compact widget for main window
- `PointCloudVisualizer`: Text-based console visualization

### 3. Enhanced Labels in Live Feed
Objects now display multi-line labels with:
- Object ID and class name
- XYZ coordinates in meters
- Azimuth and elevation angles in degrees

## How It Works:

### 3D Position Calculation:
1. **Center Point**: Finds center of bounding box in pixel coordinates
2. **Depth Value**: Uses robust depth calculation from center region of bbox
3. **3D Projection**: Uses pinhole camera model with intrinsic parameters:
   - X = (pixel_x - cx) * z / fx
   - Y = (pixel_y - cy) * z / fy  
   - Z = depth_in_meters

### Angle Calculation:
1. **Azimuth**: Horizontal angle from camera center (atan2(x, z))
   - Positive = Right side of camera
   - Negative = Left side of camera
2. **Elevation**: Vertical angle from camera center (atan2(-y, z))  
   - Positive = Above camera center
   - Negative = Below camera center

## Coordinate System:
- **X-axis**: Right is positive (meters)
- **Y-axis**: Down is positive (camera frame, meters)
- **Z-axis**: Forward is positive (meters)
- **Azimuth**: Right is positive (degrees)
- **Elevation**: Up is positive (degrees)

## Usage:

### In Code:
```python
# Get enhanced detection metadata
frames = camera_controller.get_latest_frames()
if frames and 'detection_metadata' in frames:
    detections = frames['detection_metadata']['detections']
    for detection in detections:
        if detection['position_3d']['valid']:
            pos = detection['position_3d']
            print(f"Object {detection['class_name']}:")
            print(f"  Position: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) meters")
            print(f"  Angles: Az={pos['azimuth']:.1f}°, El={pos['elevation']:.1f}°")
```

### In GUI:
- Main video feed shows enhanced labels with 3D info
- "3D Position Info" panel shows compact view
- Separate 3D visualizer window available for detailed analysis

## Technical Details:

### Camera Intrinsics:
The system automatically gets camera intrinsic parameters from the RealSense device:
- fx, fy: Focal lengths in pixels
- cx, cy: Principal point coordinates
- Default values provided for D435i if calibration fails

### Depth Processing:
- Uses center 60% region of bounding box for robust depth calculation
- Removes statistical outliers (beyond 2 standard deviations)
- Converts depth from millimeters to meters

### Label Enhancement:
Object labels now show:
```
#ID ObjectName
XYZ(x.xx, y.yy, z.zz)m
Az:angle° El:angle°
```

## Files Modified:
- `object_detection_processor.py`: Added 3D calculations
- `camera_controller.py`: Enhanced to pass camera intrinsics
- `live_3d_visualizer.py`: Complete 3D visualization widgets
- `point_cloud_visualizer.py`: Additional visualization utility

## Testing:
Run `test_3d_positioning.py` to verify calculations with sample data.

The system provides real-time 3D spatial awareness for detected objects,
enabling applications like robotics, augmented reality, and spatial analysis.
"""
