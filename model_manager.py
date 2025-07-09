"""
Model Manager for YOLO Object Detection
Handles model file selection, validation, and user prompts
"""

import os
from typing import Optional, Tuple
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class ModelManager:
    """Manages YOLO model file selection and validation"""
    
    DEFAULT_MODEL_PATH = "PredictModel/best PT 3500.pt"
    SUPPORTED_EXTENSIONS = [".pt", ".onnx", ".engine"]
    
    def __init__(self):
        self.current_model_path = None
    
    def find_model_file(self) -> Optional[str]:
        """
        Find a valid model file, starting with default path
        Returns the path if found, None otherwise
        """
        # Check default path first
        if os.path.exists(self.DEFAULT_MODEL_PATH):
            return self.DEFAULT_MODEL_PATH
        
        # Check if there are any .pt files in PredictModel directory
        predict_model_dir = "PredictModel"
        if os.path.exists(predict_model_dir):
            for file in os.listdir(predict_model_dir):
                if file.endswith('.pt'):
                    full_path = os.path.join(predict_model_dir, file)
                    if os.path.isfile(full_path):
                        return full_path
        
        # Check current directory for .pt files
        for file in os.listdir('.'):
            if file.endswith('.pt') and os.path.isfile(file):
                return file
        
        return None
    
    def validate_model_file(self, model_path: str) -> Tuple[bool, str]:
        """
        Validate if a model file is valid
        Returns (is_valid, error_message)
        """
        if not model_path:
            return False, "No model file specified"
        
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
        
        if not os.path.isfile(model_path):
            return False, f"Path is not a file: {model_path}"
        
        # Check file extension
        _, ext = os.path.splitext(model_path.lower())
        if ext not in self.SUPPORTED_EXTENSIONS:
            return False, f"Unsupported model format: {ext}. Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
        
        # Check file size (should be at least 1MB for a valid model)
        try:
            file_size = os.path.getsize(model_path)
            if file_size < 1024 * 1024:  # Less than 1MB
                return False, f"Model file seems too small ({file_size} bytes). May be corrupted."
        except OSError as e:
            return False, f"Cannot read model file: {e}"
        
        return True, "Model file is valid"
    
    def prompt_model_selection(self, parent=None) -> Optional[str]:
        """
        Show user dialog to select model file if default is not found
        Returns selected model path or None if cancelled
        """
        # First, try to find any model automatically
        auto_found = self.find_model_file()
        if auto_found:
            # Ask user if they want to use the found model
            reply = QMessageBox.question(
                parent,
                "Model File Found",
                f"Found model file:\n{auto_found}\n\nWould you like to use this model?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                return auto_found
            elif reply == QMessageBox.Cancel:
                return None
            # If No, continue to file dialog
        
        # Show manual selection dialog
        message = (
            "YOLO Model File Required\n\n"
            f"The default model file was not found:\n{self.DEFAULT_MODEL_PATH}\n\n"
            "Please select your trained YOLO model file (.pt format).\n"
            "This should be the 'best.pt' or similar file from your training."
        )
        
        reply = QMessageBox.question(
            parent,
            "Select Model File",
            message,
            QMessageBox.Ok | QMessageBox.Cancel
        )
        
        if reply != QMessageBox.Ok:
            return None
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            parent,
            "Select YOLO Model File",
            "",
            "YOLO Models (*.pt);;ONNX Models (*.onnx);;TensorRT Models (*.engine);;All Files (*)"
        )
        
        return file_path if file_path else None
    
    def get_model_info(self, model_path: str) -> dict:
        """
        Get information about a model file
        Returns dictionary with model details
        """
        if not os.path.exists(model_path):
            return {"error": "File not found"}
        
        try:
            file_size = os.path.getsize(model_path)
            file_size_mb = file_size / (1024 * 1024)
            
            return {
                "path": model_path,
                "filename": os.path.basename(model_path),
                "size_bytes": file_size,
                "size_mb": round(file_size_mb, 2),
                "format": os.path.splitext(model_path)[1],
                "exists": True
            }
        except Exception as e:
            return {"error": str(e)}
    
    def show_model_not_found_error(self, parent=None):
        """Show error message when no valid model is found"""
        message = (
            "No YOLO Model Found\n\n"
            "Object detection requires a trained YOLO model file.\n\n"
            "Expected location:\n"
            f"{self.DEFAULT_MODEL_PATH}\n\n"
            "Please ensure you have:\n"
            "• A trained YOLO model file (*.pt format)\n"
            "• Placed it in the correct directory\n"
            "• Or select a different model file when prompted\n\n"
            "You can train a model using the YOLOv11 framework or download a pre-trained model."
        )
        
        QMessageBox.critical(
            parent,
            "Model File Missing",
            message
        )
    
    def set_model_path(self, model_path: str) -> bool:
        """
        Set the current model path after validation
        Returns True if valid and set successfully
        """
        is_valid, error_msg = self.validate_model_file(model_path)
        if is_valid:
            self.current_model_path = model_path
            return True
        return False
    
    def get_current_model_path(self) -> Optional[str]:
        """Get the currently set model path"""
        return self.current_model_path
    
    def reset_model_path(self):
        """Reset the current model path"""
        self.current_model_path = None 