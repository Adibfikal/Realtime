# Custom ROS message for multiple detected objects
# This message contains a list of detected objects with header information

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