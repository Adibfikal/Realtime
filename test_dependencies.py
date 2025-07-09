#!/usr/bin/env python3
"""
Dependency Test Script for Intel RealSense D435i Live Viewer

This script tests if all required dependencies are properly installed.
Run this before launching the main application to identify any missing packages.
"""

import sys
import importlib

def test_import(module_name, package_name=None, min_version=None):
    """Test if a module can be imported"""
    display_name = package_name if package_name else module_name
    
    try:
        module = importlib.import_module(module_name)
        
        # Check version if specified
        if min_version and hasattr(module, '__version__'):
            version = module.__version__
            print(f"✅ {display_name}: {version}")
            
            # Simple version comparison (works for most cases)
            if version < min_version:
                print(f"⚠️  Warning: {display_name} version {version} is below recommended {min_version}")
        else:
            print(f"✅ {display_name}: Available")
            
        return True
        
    except ImportError as e:
        print(f"❌ {display_name}: Missing - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {display_name}: Import warning - {e}")
        return True

def test_realsense_camera():
    """Test RealSense camera availability"""
    try:
        import pyrealsense2 as rs
        
        # Try to enumerate devices
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("⚠️  RealSense Camera: No devices detected")
            print("   Make sure your D435i is connected to a USB 3.0 port")
            return False
        else:
            for i, device in enumerate(devices):
                try:
                    name = device.get_info(rs.camera_info.name)
                    serial = device.get_info(rs.camera_info.serial_number)
                    fw_version = device.get_info(rs.camera_info.firmware_version)
                    print(f"✅ RealSense Device {i}: {name}")
                    print(f"   Serial: {serial}")
                    print(f"   Firmware: {fw_version}")
                except Exception as e:
                    print(f"✅ RealSense Device {i}: Detected (info unavailable: {e})")
            return True
            
    except Exception as e:
        print(f"❌ RealSense Camera Test Failed: {e}")
        return False

def test_config_file():
    """Test if config.json exists and is valid"""
    try:
        import json
        import os
        
        if not os.path.exists("config.json"):
            print("❌ config.json: File not found")
            return False
            
        with open("config.json", 'r') as f:
            config = json.load(f)
            
        required_sections = ["device", "parameters", "viewer"]
        for section in required_sections:
            if section not in config:
                print(f"❌ config.json: Missing section '{section}'")
                return False
                
        print("✅ config.json: Valid configuration file")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ config.json: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"❌ config.json: Error reading file - {e}")
        return False

def main():
    """Main test function"""
    print("🔍 Testing Intel RealSense D435i Live Viewer Dependencies")
    print("=" * 60)
    
    # Test core Python packages
    print("\n📦 Testing Python Packages:")
    
    dependencies = [
        ("PyQt5", "PyQt5", "5.15.0"),
        ("cv2", "opencv-python", "4.5.0"),
        ("numpy", "numpy", "1.19.0"),
        ("pyrealsense2", "pyrealsense2", "2.50.0"),
        ("PIL", "Pillow", "8.0.0"),
    ]
    
    all_packages_ok = True
    for module, package, min_ver in dependencies:
        if not test_import(module, package, min_ver):
            all_packages_ok = False
    
    # Test custom modules
    print("\n🔧 Testing Custom Modules:")
    custom_modules = [
        "config_loader",
        "camera_controller"
    ]
    
    for module in custom_modules:
        test_import(module)
    
    # Test configuration file
    print("\n⚙️ Testing Configuration:")
    config_ok = test_config_file()
    
    # Test camera hardware
    print("\n📹 Testing Camera Hardware:")
    camera_ok = test_realsense_camera()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    
    if all_packages_ok:
        print("✅ All Python packages are installed correctly")
    else:
        print("❌ Some Python packages are missing or outdated")
        print("   Run: pip install -r requirements.txt")
    
    if config_ok:
        print("✅ Configuration file is valid")
    else:
        print("❌ Configuration file has issues")
    
    if camera_ok:
        print("✅ RealSense camera detected and ready")
    else:
        print("⚠️  RealSense camera not detected or has issues")
        print("   Ensure camera is connected and drivers are installed")
    
    if all_packages_ok and config_ok:
        print("\n🚀 Ready to launch application!")
        print("   Run: python main.py")
    else:
        print("\n🔧 Please resolve the issues above before running the application")
    
    return all_packages_ok and config_ok and camera_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 