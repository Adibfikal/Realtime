#!/usr/bin/env python3
"""
Test script for enhanced object detection with 3D positioning
Tests the new functionality to calculate x, y, z coordinates and angles
"""

import sys
import os
import cv2
import numpy as np
import time
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import ConfigLoader
from camera_controller import CameraController

def test_3d_detection():
    """Test the enhanced 3D detection functionality"""
    print("🚀 Testing Enhanced Object Detection with 3D Positioning")
    print("=" * 60)
    
    # Load configuration
    config_loader = ConfigLoader()
    if not config_loader.is_valid():
        print("❌ Configuration not valid")
        return False
    
    # Initialize camera controller
    camera_controller = CameraController(config_loader)
    
    # Setup object detection
    model_path = "PredictModel/best PT 3500.pt"
    if os.path.exists(model_path):
        success, message = camera_controller.setup_object_detection(model_path)
        if success:
            print(f"✓ Object detection setup: {message}")
            camera_controller.enable_detection(True)
        else:
            print(f"❌ Object detection setup failed: {message}")
            return False
    else:
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Connect to camera
    if not camera_controller.connect():
        print("❌ Failed to connect to camera")
        return False
    
    # Start streaming
    if not camera_controller.start_streaming():
        print("❌ Failed to start streaming")
        return False
    
    print("\n✓ Camera connected and streaming started")
    print("📹 Processing frames with 3D detection for 10 seconds...")
    print("Press Ctrl+C to stop early\n")
    
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10.0:  # Run for 10 seconds
            # Get latest frame
            frame_data = camera_controller.get_latest_frames()
            
            if frame_data and frame_data['detection_metadata']:
                frame_count += 1
                metadata = frame_data['detection_metadata']
                
                if 'detections' in metadata and len(metadata['detections']) > 0:
                    detection_count += 1
                    detections = metadata['detections']
                    
                    print(f"\n--- Frame {frame_count} ---")
                    print(f"Processing time: {metadata.get('processing_time_ms', 0):.1f}ms")
                    print(f"Objects detected: {len(detections)}")
                    
                    for i, detection in enumerate(detections):
                        class_name = detection.get('class_name', 'Unknown')
                        tracker_id = detection.get('tracker_id', 'N/A')
                        confidence = detection.get('confidence', 0)
                        
                        print(f"\n  Object {i+1}: {class_name} (ID: {tracker_id}, Conf: {confidence:.2f})")
                        
                        # Show basic depth info
                        depth_m = detection.get('depth_m', 0)
                        if depth_m > 0:
                            print(f"    Depth: {depth_m:.3f}m")
                        
                        # Show 3D position and angles if available
                        if 'position_3d' in detection:
                            pos_3d = detection['position_3d']
                            if pos_3d.get('valid', False):
                                print(f"    3D Position:")
                                print(f"      X: {pos_3d['x']:.3f}m (right is positive)")
                                print(f"      Y: {pos_3d['y']:.3f}m (down is positive)")
                                print(f"      Z: {pos_3d['z']:.3f}m (forward is positive)")
                                print(f"    Angles:")
                                print(f"      Azimuth: {pos_3d['azimuth']:.1f}° (horizontal)")
                                print(f"      Elevation: {pos_3d['elevation']:.1f}° (vertical)")
                            else:
                                print(f"    3D Position: Invalid (insufficient depth data)")
                    
                    # Save frame for visual inspection if needed
                    if frame_count == 1:  # Save first detection frame
                        if 'annotated' in frame_data:
                            # Convert RGB to BGR for saving
                            annotated_bgr = cv2.cvtColor(frame_data['annotated'], cv2.COLOR_RGB2BGR)
                            cv2.imwrite("test_3d_detection_frame.jpg", annotated_bgr)
                            print(f"\n📷 Saved annotated frame as 'test_3d_detection_frame.jpg'")
                
                # Brief pause to avoid flooding output
                time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    
    finally:
        # Cleanup
        camera_controller.stop_streaming()
        camera_controller.disconnect()
        
        # Show summary
        duration = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"📊 Test Summary:")
        print(f"Duration: {duration:.1f}s")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with detections: {detection_count}")
        if frame_count > 0:
            print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
        
        # Show detection stats
        stats = camera_controller.get_detection_stats()
        print(f"\n🔍 Detection Statistics:")
        print(f"Model loaded: {stats.get('model_loaded', False)}")
        print(f"Average processing time: {stats.get('avg_processing_time', 0):.1f}ms")
        print(f"Last detection count: {stats.get('last_detection_count', 0)}")
    
    print("\n✅ Test completed successfully!")
    return True

def test_angle_calculations():
    """Test angle calculation functions with known coordinates"""
    print("\n🧮 Testing angle calculation functions...")
    
    from object_detection_processor import ObjectDetectionProcessor
    processor = ObjectDetectionProcessor()
    
    # Test with known camera intrinsics
    intrinsics = {
        'fx': 616.4, 'fy': 616.8,
        'cx': 320.0, 'cy': 240.0
    }
    
    # Test cases: [pixel_x, pixel_y, depth_mm] -> expected [azimuth, elevation] approximately
    test_cases = [
        # Center of image should give 0° azimuth and elevation
        ([320, 240, 320, 240], 1000, [0, 0]),
        # Right side should give positive azimuth
        ([420, 240, 520, 240], 1000, [15, 0]),
        # Left side should give negative azimuth  
        ([220, 240, 320, 240], 1000, [-15, 0]),
        # Top should give negative elevation (up)
        ([320, 140, 320, 240], 1000, [0, -15]),
        # Bottom should give positive elevation (down)
        ([320, 340, 320, 440], 1000, [0, 15]),
    ]
    
    for i, (bbox, depth, expected) in enumerate(test_cases):
        result = processor.calculate_3d_position_and_angles(bbox, depth, intrinsics)
        
        if result['valid']:
            azimuth = result['azimuth']
            elevation = result['elevation']
            print(f"  Test {i+1}: Az={azimuth:.1f}° El={elevation:.1f}° "
                  f"(Expected: Az≈{expected[0]}° El≈{expected[1]}°)")
        else:
            print(f"  Test {i+1}: Invalid result")
    
    print("✅ Angle calculation tests completed")

if __name__ == "__main__":
    print("🔧 Enhanced Object Detection 3D Test")
    print("This test will verify the new 3D positioning functionality")
    print("Make sure the camera is connected and the model file exists.\n")
    
    # Test angle calculations first
    test_angle_calculations()
    
    # Ask user if they want to run live camera test
    response = input("\nRun live camera test? (y/n): ").lower().strip()
    if response.startswith('y'):
        test_3d_detection()
    else:
        print("Skipping live camera test.")
        print("✅ Tests completed!")
