"""
Test script to verify 3D position calculation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from object_detection_processor import ObjectDetectionProcessor

def test_3d_position_calculation():
    """Test the 3D position and angle calculations"""
    
    # Create processor instance
    processor = ObjectDetectionProcessor()
    
    # Test camera intrinsics (D435i default)
    camera_intrinsics = {
        'fx': 616.4,
        'fy': 616.8,
        'cx': 320.0,
        'cy': 240.0
    }
    
    print("Testing 3D Position and Angle Calculations")
    print("=" * 50)
    
    # Test cases: [x1, y1, x2, y2], depth_mm
    test_cases = [
        # Center object
        ([300, 220, 340, 260], 1000),  # Center, 1m away
        
        # Right side object
        ([500, 220, 540, 260], 1500),  # Right side, 1.5m away
        
        # Left side object  
        ([100, 220, 140, 260], 2000),  # Left side, 2m away
        
        # Upper object
        ([300, 100, 340, 140], 1200),  # Above center, 1.2m away
        
        # Lower object
        ([300, 350, 340, 390], 800),   # Below center, 0.8m away
    ]
    
    for i, (bbox, depth) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Bbox: {bbox}, Depth: {depth}mm")
        
        result = processor.calculate_3d_position_and_angles(bbox, depth, camera_intrinsics)
        
        if result['valid']:
            print(f"3D Position: X={result['x']:.3f}m, Y={result['y']:.3f}m, Z={result['z']:.3f}m")
            print(f"Angles: Azimuth={result['azimuth']:.1f}°, Elevation={result['elevation']:.1f}°")
            
            # Interpret the results
            if result['azimuth'] > 5:
                side = "Right"
            elif result['azimuth'] < -5:
                side = "Left"
            else:
                side = "Center"
                
            if result['elevation'] > 5:
                vertical = "Above"
            elif result['elevation'] < -5:
                vertical = "Below"
            else:
                vertical = "Level with"
            
            print(f"Interpretation: {side} side, {vertical} camera center")
        else:
            print("Invalid position data")
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")

if __name__ == "__main__":
    test_3d_position_calculation()
