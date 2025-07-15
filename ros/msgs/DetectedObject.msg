# Custom ROS message for detected objects in SLAM/Navigation
# This message contains all necessary information for a detected object

# Object identification
int32 object_id
string class_name
float32 confidence
time timestamp

# 2D bounding box in image coordinates (pixels)
float32[4] bbox_2d  # [x1, y1, x2, y2]
float32[2] image_center  # [x, y] center of bbox

# 3D position in camera frame (meters)
geometry_msgs/Point position_3d
float32 depth_mm

# Camera information
string camera_frame
int32 image_width
int32 image_height

# SLAM-specific information
bool is_static
float32 reliability_score
float32[3] position_uncertainty  # [x, y, z] uncertainty in meters 