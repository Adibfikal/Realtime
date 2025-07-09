# Intel RealSense D435i Live Viewer - Installation Guide

## 🎯 Quick Setup (Recommended)

### Prerequisites
- Windows 10/11 with USB 3.0 port
- Intel RealSense D435i camera
- Anaconda or Miniconda installed

### Step 1: Create Conda Environment
```bash
# Create a new conda environment with Python 3.7
conda create -n realsense python=3.7
conda activate realsense
```

### Step 2: Install Dependencies
```bash
# Install packages via conda (recommended for Windows)
conda install -c conda-forge pyqt pillow

# Install RealSense and OpenCV via pip
pip install pyrealsense2 opencv-python numpy
```

### Step 3: Verify Installation
```bash
# Test all dependencies
python test_dependencies.py
```

### Step 4: Launch Application
```bash
# Start the GUI application
python main.py
```

## 🔧 Alternative Setup Methods

### Method 1: Using requirements.txt (may require Visual C++)
```bash
pip install -r requirements.txt
```

### Method 2: Manual Installation
```bash
# Core packages
conda install -c conda-forge pyqt=5.15.7 pillow=9.2.0

# Computer vision packages
pip install pyrealsense2>=2.54.0
pip install opencv-python>=4.5.0
pip install numpy>=1.19.0
```

## 🐛 Troubleshooting

### Issue: "Microsoft Visual C++ 14.0 or greater is required"
**Solution**: Use conda instead of pip for PyQt5:
```bash
conda install -c conda-forge pyqt
```

### Issue: "No RealSense devices detected"
**Solutions**:
1. Check USB 3.0 connection
2. Install Intel RealSense SDK
3. Try different USB port
4. Restart camera/computer

### Issue: Import errors
**Solution**: Verify conda environment is activated:
```bash
conda activate realsense
which python  # Should show conda environment path
```

## ✅ Verified Working Environment
- **OS**: Windows 10/11 64-bit
- **Python**: 3.7.12-3.7.16
- **PyQt**: 5.15.7 (conda-forge)
- **pyrealsense2**: 2.55.1.6486
- **opencv-python**: 4.11.0.86
- **numpy**: 1.21.6
- **pillow**: 9.2.0

## 📋 Environment Summary
After successful installation, your environment should contain:
```
conda list
# Should show: pyqt, pillow, pyrealsense2, opencv-python, numpy
```

## 🚀 Next Steps
1. Connect your D435i camera
2. Run `python test_dependencies.py` to verify setup
3. Launch the application with `python main.py`
4. Use "Connect Camera" button in the GUI 