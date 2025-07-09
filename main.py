#!/usr/bin/env python3
"""
Intel RealSense D435i Live Viewer Application

A PyQt5-based GUI application for viewing live RGB and depth streams
from Intel RealSense D435i camera with recording capabilities.

Author: Assistant
Date: 2024
"""

import sys
import os

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main_gui import main
    
    if __name__ == "__main__":
        print("🚀 Starting Intel RealSense D435i Live Viewer...")
        print("📋 Loading configuration and initializing GUI...")
        main()
        
except ImportError as e:
    print("❌ Import Error: Missing required dependencies")
    print(f"Error details: {e}")
    print("\n📦 Please install required packages:")
    print("pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Application Error: {e}")
    sys.exit(1) 