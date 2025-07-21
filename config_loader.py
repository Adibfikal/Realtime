import json
import os
from typing import Dict, Any, Optional

class ConfigLoader:
    """Loads and manages Intel RealSense camera configuration"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from JSON file"""
        try:
            if not os.path.exists(self.config_path):
                print(f"Warning: Config file {self.config_path} not found")
                return False
                
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            print(f"✓ Configuration loaded successfully from {self.config_path}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return self.config.get("device", {})
    
    def get_stream_config(self) -> Dict[str, Any]:
        """Get stream configuration (resolution, fps, format)"""
        viewer = self.config.get("viewer", {})
        return {
            "width": int(viewer.get("stream-width", 640)),
            "height": int(viewer.get("stream-height", 480)),
            "fps": int(viewer.get("stream-fps", 30)),
            "depth_format": viewer.get("stream-depth-format", "Z16")
        }
    
    def get_camera_parameters(self) -> Dict[str, Any]:
        """Get camera control parameters"""
        params = self.config.get("parameters", {})
        
        # Extract key parameters for camera setup
        camera_params = {}
        
        # Auto exposure settings
        if "controls-autoexposure-auto" in params:
            camera_params["auto_exposure"] = params["controls-autoexposure-auto"].lower() == "true"
        if "controls-autoexposure-manual" in params:
            camera_params["exposure"] = int(params["controls-autoexposure-manual"])
            
        # Color settings
        if "controls-color-autoexposure-auto" in params:
            camera_params["color_auto_exposure"] = params["controls-color-autoexposure-auto"].lower() == "true"
        if "controls-color-autoexposure-manual" in params:
            camera_params["color_exposure"] = int(params["controls-color-autoexposure-manual"])
        if "controls-color-gain" in params:
            camera_params["color_gain"] = int(params["controls-color-gain"])
        if "controls-color-white-balance-auto" in params:
            camera_params["auto_white_balance"] = params["controls-color-white-balance-auto"].lower() == "true"
        if "controls-color-white-balance-manual" in params:
            camera_params["white_balance"] = int(params["controls-color-white-balance-manual"])
            
        # Depth settings
        if "controls-depth-gain" in params:
            camera_params["depth_gain"] = int(params["controls-depth-gain"])
        if "controls-laserpower" in params:
            camera_params["laser_power"] = int(params["controls-laserpower"])
        if "controls-laserstate" in params:
            camera_params["laser_enabled"] = params["controls-laserstate"].lower() == "on"
            
        # Depth processing parameters
        if "param-depthunits" in params:
            camera_params["depth_units"] = int(params["param-depthunits"])
        if "param-depthclampmin" in params:
            camera_params["depth_clamp_min"] = int(params["param-depthclampmin"])
        if "param-depthclampmax" in params:
            camera_params["depth_clamp_max"] = int(params["param-depthclampmax"])
            
        return camera_params
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all raw parameters from config"""
        return self.config.get("parameters", {})
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return bool(self.config and "device" in self.config)
    
    def print_summary(self):
        """Print configuration summary"""
        if not self.is_valid():
            print("❌ Invalid configuration")
            return
            
        device_info = self.get_device_info()
        stream_config = self.get_stream_config()
        
        print("📹 Camera Configuration Summary:")
        print(f"  Device: {device_info.get('name', 'Unknown')}")
        print(f"  Product Line: {device_info.get('product line', 'Unknown')}")
        print(f"  Firmware: {device_info.get('fw version', 'Unknown')}")
        print(f"  Resolution: {stream_config['width']}x{stream_config['height']}")
        print(f"  FPS: {stream_config['fps']}")
    def get_enhanced_detection_config(self) -> Dict[str, Any]:
        """Get enhanced detection configuration settings"""
        enhanced_config = self.config.get("enhanced_detection", {})
        
        # Default configuration
        default_config = {
            "point_cloud_processing": {
                "weighting_method": "combined",
                "distance_weight": 0.3,
                "validity_weight": 0.3,
                "statistical_weight": 0.2,
                "spatial_weight": 0.2,
                "min_valid_points": 10,
                "max_processing_region": 0.8,
                "outlier_threshold": 2.0,
                "neighborhood_window_size": 3
            },
            "angle_calculation": {
                "coordinate_system": "camera_centered",
                "angle_units": "degrees",
                "precision": 1,
                "enable_azimuth": True,
                "enable_elevation": True
            },
            "visualization": {
                "show_coordinates": True,
                "show_angles": True,
                "label_format": "detailed",
                "coordinate_precision": 2,
                "angle_precision": 1,
                "compact_display": False,
                "enable_quality_indicators": True
            },
            "camera_intrinsics": {
                "auto_detect": True,
                "fallback_fx": 525.0,
                "fallback_fy": 525.0,
                "fallback_cx": 320.0,
                "fallback_cy": 240.0,
                "update_frequency_frames": 30
            },
            "performance": {
                "enable_gpu_acceleration": True,
                "enable_half_precision": True,
                "max_processing_time_ms": 50,
                "enable_profiling": False
            },
            "quality_control": {
                "min_point_confidence": 0.3,
                "min_depth_quality": 0.2,
                "enable_outlier_filtering": True,
                "enable_consistency_check": True
            }
        }
        
        # Merge with loaded configuration
        for category, settings in default_config.items():
            if category in enhanced_config:
                settings.update(enhanced_config[category])
            enhanced_config[category] = settings
        
        return enhanced_config
    
    def get_point_cloud_config(self) -> Dict[str, Any]:
        """Get point cloud processing configuration"""
        enhanced_config = self.get_enhanced_detection_config()
        return enhanced_config.get("point_cloud_processing", {})

    def get_angle_calculation_config(self) -> Dict[str, Any]:
        """Get angle calculation configuration"""
        enhanced_config = self.get_enhanced_detection_config()
        return enhanced_config.get("angle_calculation", {})

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        enhanced_config = self.get_enhanced_detection_config()
        return enhanced_config.get("visualization", {})

    def get_camera_intrinsics_config(self) -> Dict[str, Any]:
        """Get camera intrinsics configuration"""
        enhanced_config = self.get_enhanced_detection_config()
        return enhanced_config.get("camera_intrinsics", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        enhanced_config = self.get_enhanced_detection_config()
        return enhanced_config.get("performance", {})

    def get_quality_control_config(self) -> Dict[str, Any]:
        """Get quality control configuration"""
        enhanced_config = self.get_enhanced_detection_config()
        return enhanced_config.get("quality_control", {})

    def update_enhanced_config(self, category: str, updates: Dict[str, Any]) -> bool:
        """Update enhanced detection configuration"""
        try:
            if "enhanced_detection" not in self.config:
                self.config["enhanced_detection"] = {}
            
            if category not in self.config["enhanced_detection"]:
                self.config["enhanced_detection"][category] = {}
            
            self.config["enhanced_detection"][category].update(updates)
            
            print(f"✓ Enhanced detection config updated for {category}")
            return True
            
        except Exception as e:
            print(f"Error updating enhanced config: {e}")
            return False

    def is_enhanced_detection_enabled(self) -> bool:
        """Check if enhanced detection features are enabled"""
        return "enhanced_detection" in self.config

    def save_config(self, output_file: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        try:
            output_path = output_file or self.config_path
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            print(f"✓ Configuration saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving config: {e}")
            return False