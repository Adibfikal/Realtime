# Custom ROS message for detected objects in SLAM/Navigation
# This message contains all necessary information for a detected object
# Enhanced version 2.0 with weighted center point and angle calculations

# Object identification
int32 object_id
string class_name
float32 confidence
time timestamp

# 2D bounding box in image coordinates (pixels)
float32[4] bbox_2d  # [x1, y1, x2, y2]
float32[2] image_center  # [x, y] center of bbox

# Enhanced 3D information with weighted center point
geometry_msgs/Point position_3d  # Original 3D position
geometry_msgs/Point center_point_3d  # Weighted center point in 3D
float32[2] center_point_image  # Center point in image coordinates
float32 depth_mm

# Spherical coordinate angles (degrees)
float32 azimuth_deg  # Azimuth angle in degrees (-180 to +180)
float32 elevation_deg  # Elevation angle in degrees (-90 to +90)

# Quality and confidence metrics
float32 point_confidence  # Confidence of center point calculation (0.0-1.0)
float32 depth_quality  # Quality score of depth measurement (0.0-1.0)
float32[4] depth_statistics  # [mean, std, min, max] of depth region in mm

# Camera information
string camera_frame
int32 image_width
int32 image_height

# Camera intrinsics used for calculations
float32 focal_length_x  # fx
float32 focal_length_y  # fy
float32 principal_point_x  # cx
float32 principal_point_y  # cy

# SLAM-specific information
bool is_static
float32 reliability_score
float32[3] position_uncertainty  # [x, y, z] uncertainty in meters

# Processing metadata
float32 processing_time_ms  # Time taken for enhanced processing
string processing_version  # Version identifier for processing algorithm