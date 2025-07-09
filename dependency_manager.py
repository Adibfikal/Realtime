"""
Dependency Manager for Object Detection Requirements
Handles checking and installing required packages with user prompts
"""

import subprocess
import sys
from typing import List, Tuple
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QApplication
from PyQt5.QtCore import QThread, pyqtSignal, Qt

class PackageInstaller(QThread):
    """Thread for installing packages without blocking the UI"""
    
    progress = pyqtSignal(str)  # Progress message
    finished = pyqtSignal(bool, str)  # Success flag and message
    
    def __init__(self, packages: List[str]):
        super().__init__()
        self.packages = packages
    
    def run(self):
        """Install packages using pip"""
        try:
            for package in self.packages:
                self.progress.emit(f"Installing {package}...")
                
                # Run pip install
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per package
                )
                
                if result.returncode != 0:
                    error_msg = f"Failed to install {package}: {result.stderr}"
                    self.finished.emit(False, error_msg)
                    return
            
            self.finished.emit(True, "All packages installed successfully!")
            
        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Installation timed out. Please try manually.")
        except Exception as e:
            self.finished.emit(False, f"Installation error: {str(e)}")

class DependencyManager:
    """Manages detection dependencies and provides installation prompts"""
    
    DETECTION_PACKAGES = [
        "ultralytics>=8.0.0",
        "supervision>=0.16.0", 
        "torch>=1.9.0",
        "torchvision>=0.10.0"
    ]
    
    def __init__(self):
        self.installer_thread = None
    
    def check_detection_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Check if detection dependencies are available
        Returns (all_available, missing_packages)
        """
        missing_packages = []
        
        # Check ultralytics
        try:
            __import__("ultralytics")
        except ImportError:
            missing_packages.append("ultralytics")
        
        # Check supervision
        try:
            __import__("supervision")
        except ImportError:
            missing_packages.append("supervision")
        
        # Check torch
        try:
            __import__("torch")
        except ImportError:
            missing_packages.append("torch")
        
        # Check torchvision
        try:
            __import__("torchvision")
        except ImportError:
            missing_packages.append("torchvision")
        
        return len(missing_packages) == 0, missing_packages
    
    def prompt_install_dependencies(self, parent=None) -> bool:
        """
        Show user dialog to install missing dependencies
        Returns True if user agrees to install
        """
        available, missing = self.check_detection_dependencies()
        
        if available:
            return True
        
        # Create informative message
        message = (
            "Object Detection Dependencies Missing\n\n"
            f"The following packages are required for object detection:\n"
            f"• {', '.join(missing)}\n\n"
            "These packages are needed to run YOLO object detection with depth information.\n\n"
            "Would you like to install them now?\n"
            "(This may take a few minutes)"
        )
        
        reply = QMessageBox.question(
            parent,
            "Install Dependencies",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        return reply == QMessageBox.Yes
    
    def install_dependencies_with_progress(self, parent=None) -> Tuple[bool, str]:
        """
        Install dependencies with progress dialog
        Returns (success, message)
        """
        # Create progress dialog
        progress_dialog = QProgressDialog(
            "Installing object detection dependencies...",
            "Cancel",
            0, 0,  # Indeterminate progress
            parent
        )
        progress_dialog.setWindowTitle("Installing Dependencies")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setCancelButton(None)  # Don't allow cancellation
        progress_dialog.show()
        
        # Prepare packages to install
        packages_to_install = []
        available, missing = self.check_detection_dependencies()
        
        if "ultralytics" in missing:
            packages_to_install.append("ultralytics>=8.0.0")
        if "supervision" in missing:
            packages_to_install.append("supervision>=0.16.0")
        if "torch" in missing:
            packages_to_install.append("torch>=1.9.0")
        if "torchvision" in missing:
            packages_to_install.append("torchvision>=0.10.0")
        
        if not packages_to_install:
            progress_dialog.close()
            return True, "All dependencies already installed"
        
        # Start installation thread
        self.installer_thread = PackageInstaller(packages_to_install)
        
        # Connect signals
        def update_progress(message):
            progress_dialog.setLabelText(message)
            QApplication.processEvents()
        
        def installation_finished(success, message):
            progress_dialog.close()
            self.installation_result = (success, message)
        
        self.installer_thread.progress.connect(update_progress)
        self.installer_thread.finished.connect(installation_finished)
        
        # Start installation
        self.installation_result = None
        self.installer_thread.start()
        
        # Wait for completion
        while self.installation_result is None:
            QApplication.processEvents()
            self.installer_thread.msleep(100)
        
        success, message = self.installation_result
        
        if success:
            QMessageBox.information(
                parent,
                "Installation Complete",
                "Object detection dependencies installed successfully!\n\n"
                "Please restart the application to use object detection features."
            )
        else:
            QMessageBox.critical(
                parent,
                "Installation Failed",
                f"Failed to install dependencies:\n\n{message}\n\n"
                "Please try installing manually:\n"
                f"pip install {' '.join(packages_to_install)}"
            )
        
        return success, message
    
    def show_manual_installation_guide(self, parent=None):
        """Show manual installation instructions"""
        message = (
            "Manual Installation Instructions\n\n"
            "Please run the following commands in your terminal/command prompt:\n\n"
            f"pip install {' '.join(self.DETECTION_PACKAGES)}\n\n"
            "After installation, restart the application to use object detection features."
        )
        
        QMessageBox.information(
            parent,
            "Manual Installation",
            message
        )
    
    def get_dependency_status(self) -> dict:
        """Get detailed status of all dependencies"""
        status = {}
        
        dependencies = [
            ("ultralytics", "YOLO object detection"),
            ("supervision", "Object tracking and annotation"),
            ("torch", "PyTorch deep learning framework"),
            ("torchvision", "Computer vision utilities")
        ]
        
        for package, description in dependencies:
            try:
                __import__(package)
                status[package] = {"available": True, "description": description}
            except ImportError:
                status[package] = {"available": False, "description": description}
        
        return status 