"""
Demo script showing 3D position calculation with sample data
Run this to see how the enhanced labels will look in the live feed
"""

import numpy as np
from object_detection_processor import ObjectDetectionProcessor

def demo_enhanced_labels():
    """Demonstrate the enhanced object labels with 3D positioning"""
    
    # Sample detection data (simulating real detections)
    sample_detections = [
        {
            'tracker_id': 1,
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': [300, 200, 400, 350],  # Center person
            'depth_mm': 1500
        },
        {
            'tracker_id': 2, 
            'class_name': 'bottle',
            'confidence': 0.92,
            'bbox': [480, 250, 520, 320],  # Right side bottle
            'depth_mm': 800
        },
        {
            'tracker_id': 3,
            'class_name': 'laptop',
            'confidence': 0.78,
            'bbox': [120, 180, 220, 280],  # Left side laptop
            'depth_mm': 2200
        }
    ]
    
    # Camera intrinsics
    camera_intrinsics = {
        'fx': 616.4, 'fy': 616.8,
        'cx': 320.0, 'cy': 240.0
    }
    
    processor = ObjectDetectionProcessor()
    
    print("ENHANCED OBJECT DETECTION LABELS")
    print("=" * 50)
    print("This is how objects will appear in the live feed:")
    print()
    
    for detection in sample_detections:
        # Calculate 3D position
        pos_3d = processor.calculate_3d_position_and_angles(
            detection['bbox'], detection['depth_mm'], camera_intrinsics
        )
        
        if pos_3d['valid']:
            # Generate the enhanced label (same format as in live feed)
            label = (f"#{detection['tracker_id']} {detection['class_name']}\n"
                    f"XYZ: ({pos_3d['x']:.2f}, {pos_3d['y']:.2f}, {pos_3d['z']:.2f})m\n"
                    f"Az: {pos_3d['azimuth']:.1f}° El: {pos_3d['elevation']:.1f}°")
            
            print(f"Label for {detection['class_name']}:")
            print("-" * 30)
            print(label)
            print()
            
            # Human-readable interpretation
            if pos_3d['azimuth'] > 10:
                side = "right side"
            elif pos_3d['azimuth'] < -10:
                side = "left side"
            else:
                side = "center"
                
            if pos_3d['elevation'] > 5:
                vertical = "above"
            elif pos_3d['elevation'] < -5:
                vertical = "below"
            else:
                vertical = "level with"
                
            print(f"→ Located on {side}, {vertical} camera center")
            print(f"→ Distance: {pos_3d['z']:.2f} meters")
            print("=" * 30)
            print()

if __name__ == "__main__":
    demo_enhanced_labels()
