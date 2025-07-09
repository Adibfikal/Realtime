#!/usr/bin/env python3
"""
Quick test to verify object detection setup fix
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config_loader import ConfigLoader
    from object_detection_processor import ObjectDetectionProcessor
    
    print("🔍 Testing object detection setup...")
    
    # Initialize processor
    processor = ObjectDetectionProcessor()
    
    # Test model loading
    model_path = "PredictModel/best PT 3500.pt"
    if os.path.exists(model_path):
        success, message = processor.load_model(model_path)
        if success:
            print(f"✅ {message}")
            print("🎉 Object detection setup successful!")
        else:
            print(f"❌ {message}")
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("   Testing with initialization only...")
        print("✅ Object detection processor initialized without errors")
        
except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()
