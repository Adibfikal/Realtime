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
        print(f"  Depth Format: {stream_config['depth_format']}") 