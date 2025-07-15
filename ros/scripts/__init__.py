"""
ROS Scripts Package
Contains ROS communication and receiver scripts
"""

# Only import classes when they are actually needed to avoid dependency issues
# This allows the package to be imported even when ROS is not available

__all__ = ["ROS1NoeticCommunication", "ROS1NoeticReceiver"]

# Lazy imports to avoid dependency issues
def get_ros_noetic_communication():
    """Get ROS1NoeticCommunication class (lazy import)"""
    from .ros_noetic_communication import ROS1NoeticCommunication
    return ROS1NoeticCommunication

def get_ros_noetic_receiver():
    """Get ROS1NoeticReceiver class (lazy import)"""
    from .ros_noetic_receiver import ROS1NoeticReceiver
    return ROS1NoeticReceiver 