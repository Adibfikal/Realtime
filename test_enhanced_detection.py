#!/usr/bin/env python3
"""
Comprehensive test script for enhanced object detection system
Tests weighted center point extraction, angle calculations, and integration
"""

import os
import sys
import time
import numpy as np
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from enhanced_object_detection_processor import EnhancedObjectDetectionProcessor
    from config_loader import ConfigLoader
    from communication_handler import CommunicationHandler
    print("✓ All enhanced detection modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

@dataclass
class TestResult:
    """Test result structure"""
    test_name: str
    passed: bool
    execution_time: float
    details: Dict[str, Any]
    error: str = ""

class EnhancedDetectionTester:
    """Comprehensive tester for enhanced object detection system"""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.test_results: List[TestResult] = []
        self.processor = None
        
        print("🧪 Enhanced Detection Test Suite")
        print("=" * 50)
    
    def run_all_tests(self) -> bool:
        """Run all test categories"""
        print("\n📋 Starting comprehensive test suite...")
        
        # Test categories
        test_categories = [
            ("Configuration System", self.test_configuration_system),
            ("Enhanced Processor Creation", self.test_processor_creation),
            ("Weighted Center Point Extraction", self.test_weighted_center_point),
            ("Camera Intrinsics Integration", self.test_camera_intrinsics),
            ("3D Coordinate Conversion", self.test_3d_coordinate_conversion),
            ("Spherical Angle Calculations", self.test_spherical_angles),
            ("Point Cloud Processing", self.test_point_cloud_processing),
            ("Visualization Enhancements", self.test_visualization_features),
            ("Performance Impact", self.test_performance_impact),
            ("Integration Test", self.test_system_integration)
        ]
        
        overall_success = True
        for category_name, test_func in test_categories:
            print(f"\n🔍 Testing {category_name}...")
            try:
                success = test_func()
                if not success:
                    overall_success = False
                    print(f"❌ {category_name} tests failed")
                else:
                    print(f"✓ {category_name} tests passed")
            except Exception as e:
                print(f"❌ {category_name} test error: {e}")
                overall_success = False
        
        self.print_test_summary()
        return overall_success
    
    def test_configuration_system(self) -> bool:
        """Test enhanced configuration system"""
        start_time = time.time()
        
        try:
            # Test enhanced detection config loading
            enhanced_config = self.config_loader.get_enhanced_detection_config()
            
            # Verify all required sections exist
            required_sections = [
                "point_cloud_processing", "angle_calculation", "visualization",
                "camera_intrinsics", "performance", "quality_control"
            ]
            
            for section in required_sections:
                if section not in enhanced_config:
                    raise ValueError(f"Missing configuration section: {section}")
            
            # Test specific config getters
            point_cloud_config = self.config_loader.get_point_cloud_config()
            angle_config = self.config_loader.get_angle_calculation_config()
            viz_config = self.config_loader.get_visualization_config()
            
            # Verify key parameters
            assert "weighting_method" in point_cloud_config
            assert "angle_units" in angle_config
            assert "show_coordinates" in viz_config
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Configuration System",
                True,
                execution_time,
                {"sections": len(required_sections), "config_size": len(enhanced_config)}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Configuration System",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_processor_creation(self) -> bool:
        """Test enhanced processor creation and initialization"""
        start_time = time.time()
        
        try:
            # Create enhanced processor
            self.processor = EnhancedObjectDetectionProcessor()
            
            # Verify required attributes exist
            required_attributes = [
                'POINT_CLOUD_CONFIG', 'ANGLE_CONFIG', 'DISPLAY_OPTIONS',
                'camera_intrinsics'
            ]
            
            for attr in required_attributes:
                if not hasattr(self.processor, attr):
                    raise ValueError(f"Missing processor attribute: {attr}")
            
            # Test method existence
            required_methods = [
                'calculate_weighted_center_point', 'get_camera_intrinsics',
                'image_to_3d_coordinates', 'calculate_spherical_angles'
            ]
            
            for method in required_methods:
                if not hasattr(self.processor, method):
                    raise ValueError(f"Missing processor method: {method}")
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Enhanced Processor Creation",
                True,
                execution_time,
                {
                    "attributes_count": len(required_attributes),
                    "methods_count": len(required_methods),
                    "enhanced_features": True
                }
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Enhanced Processor Creation",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_weighted_center_point(self) -> bool:
        """Test weighted center point extraction"""
        start_time = time.time()
        
        try:
            if not self.processor:
                raise ValueError("Processor not initialized")
            
            # Create test depth data
            test_depth = np.random.randint(500, 3000, (100, 100), dtype=np.uint16)
            test_bbox = [20.0, 20.0, 80.0, 80.0]  # x1, y1, x2, y2 as List[float]
            
            # Test different weighting methods
            weighting_methods = ["distance", "validity", "statistical", "spatial", "combined"]
            
            results = {}
            for method in weighting_methods:
                # Update config for this test
                self.processor.POINT_CLOUD_CONFIG["weighting_method"] = method
                
                result = self.processor.calculate_weighted_center_point(
                    test_depth, test_bbox
                )
                
                if result is not None and len(result) == 3:
                    center_point, depth_value, metadata = result
                    x, y = center_point
                    
                    # Verify reasonable coordinate ranges
                    assert 20 <= x <= 80, f"X coordinate {x} out of bbox range"
                    assert 20 <= y <= 80, f"Y coordinate {y} out of bbox range"
                    assert 500 <= depth_value <= 3000, f"Depth {depth_value} out of depth range"
                    
                    results[method] = {"x": float(x), "y": float(y), "z": float(depth_value), "metadata": metadata}
                else:
                    results[method] = None
            
            # Verify at least some methods produced valid results
            valid_results = sum(1 for v in results.values() if v is not None)
            assert valid_results >= 3, f"Too few valid results: {valid_results}/5"
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Weighted Center Point Extraction",
                True,
                execution_time,
                {"methods_tested": len(weighting_methods), "valid_results": valid_results, "results": results}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Weighted Center Point Extraction",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_camera_intrinsics(self) -> bool:
        """Test camera intrinsics integration"""
        start_time = time.time()
        
        try:
            if not self.processor:
                raise ValueError("Processor not initialized")
            
            # Test intrinsics retrieval
            intrinsics = self.processor.get_camera_intrinsics()
            
            # Verify intrinsics structure
            required_keys = ["fx", "fy", "cx", "cy"]
            for key in required_keys:
                assert key in intrinsics, f"Missing intrinsic parameter: {key}"
                assert isinstance(intrinsics[key], (int, float)), f"Invalid intrinsic type for {key}"
                assert intrinsics[key] > 0, f"Invalid intrinsic value for {key}: {intrinsics[key]}"
            
            # Test reasonable value ranges
            assert 200 <= intrinsics["fx"] <= 2000, f"Unrealistic fx: {intrinsics['fx']}"
            assert 200 <= intrinsics["fy"] <= 2000, f"Unrealistic fy: {intrinsics['fy']}"
            assert 100 <= intrinsics["cx"] <= 1000, f"Unrealistic cx: {intrinsics['cx']}"
            assert 100 <= intrinsics["cy"] <= 1000, f"Unrealistic cy: {intrinsics['cy']}"
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Camera Intrinsics Integration",
                True,
                execution_time,
                {"intrinsics": intrinsics}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Camera Intrinsics Integration",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_3d_coordinate_conversion(self) -> bool:
        """Test 3D coordinate conversion"""
        start_time = time.time()
        
        try:
            if not self.processor:
                raise ValueError("Processor not initialized")
            
            # Test coordinate conversion
            test_cases = [
                (320, 240, 1000),  # Center point
                (100, 100, 500),   # Top-left
                (500, 400, 2000),  # Bottom-right
                (0, 0, 100),       # Edge case
            ]
            
            results = []
            for u, v, depth in test_cases:
                coords_3d = self.processor.image_to_3d_coordinates((u, v), depth, self.processor.camera_intrinsics)
                
                if coords_3d is not None:
                    x, y, z = coords_3d
                    
                    # Verify reasonable 3D coordinate ranges
                    assert -5000 <= x <= 5000, f"X coordinate {x} out of range"
                    assert -5000 <= y <= 5000, f"Y coordinate {y} out of range"
                    assert abs(z - depth/1000.0) < 0.001, f"Z coordinate {z} should equal depth in meters {depth/1000.0}"
                    
                    results.append({
                        "input": (u, v, depth),
                        "output": (float(x), float(y), float(z))
                    })
                else:
                    results.append({
                        "input": (u, v, depth),
                        "output": None
                    })
            
            # Verify at least most conversions worked
            valid_conversions = sum(1 for r in results if r["output"] is not None)
            assert valid_conversions >= len(test_cases) * 0.75, f"Too few valid conversions: {valid_conversions}/{len(test_cases)}"
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "3D Coordinate Conversion",
                True,
                execution_time,
                {"test_cases": len(test_cases), "valid_conversions": valid_conversions, "results": results}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "3D Coordinate Conversion",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_spherical_angles(self) -> bool:
        """Test spherical angle calculations"""
        start_time = time.time()
        
        try:
            if not self.processor:
                raise ValueError("Processor not initialized")
            
            # Test angle calculations with known 3D coordinates
            test_cases = [
                (1000, 0, 1000),     # 45° azimuth, 0° elevation
                (0, 1000, 1000),     # 90° azimuth, 45° elevation
                (-1000, 0, 1000),    # -45° azimuth, 0° elevation
                (0, -1000, 1000),    # -90° azimuth, -45° elevation
                (0, 0, 1000),        # 0° azimuth, 0° elevation
            ]
            
            results = []
            for x, y, z in test_cases:
                angles = self.processor.calculate_spherical_angles(x, y, z)
                
                if angles is not None:
                    azimuth, elevation = angles
                    
                    # Verify angle ranges
                    assert -180 <= azimuth <= 180, f"Azimuth {azimuth} out of range"
                    assert -90 <= elevation <= 90, f"Elevation {elevation} out of range"
                    
                    results.append({
                        "input": (x, y, z),
                        "output": (float(azimuth), float(elevation))
                    })
                else:
                    results.append({
                        "input": (x, y, z),
                        "output": None
                    })
            
            # Verify expected angle calculations
            valid_calculations = sum(1 for r in results if r["output"] is not None)
            assert valid_calculations == len(test_cases), f"Invalid angle calculations: {valid_calculations}/{len(test_cases)}"
            
            # Test specific expected results (within tolerance)
            if results[0]["output"]:  # (1000, 0, 1000) should give ~45° azimuth
                azimuth = results[0]["output"][0]
                assert abs(azimuth - 45) < 5, f"Expected ~45° azimuth, got {azimuth}"
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Spherical Angle Calculations",
                True,
                execution_time,
                {"test_cases": len(test_cases), "valid_calculations": valid_calculations, "results": results}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Spherical Angle Calculations",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_point_cloud_processing(self) -> bool:
        """Test point cloud processing features"""
        start_time = time.time()
        
        try:
            if not self.processor:
                raise ValueError("Processor not initialized")
            
            # Create test depth region with some outliers
            test_region = np.random.randint(800, 1200, (50, 50), dtype=np.uint16)
            
            # Add some outliers
            test_region[5:10, 5:10] = 5000  # Far outliers
            test_region[40:45, 40:45] = 100  # Near outliers
            test_region[25, 25] = 0  # Invalid depth
            
            # Test point cloud processing by using the weighted center point calculation
            # which internally processes the point cloud region
            test_bbox = [10.0, 10.0, 40.0, 40.0]
            result = self.processor.calculate_weighted_center_point(test_region, test_bbox)
            
            # Verify processing results
            assert result is not None, "Point cloud processing returned None"
            center_point, depth_value, metadata = result
            
            assert "point_confidence" in metadata, "Missing point_confidence in metadata"
            assert "depth_quality" in metadata, "Missing depth_quality in metadata"
            assert "valid_points" in metadata, "Missing valid_points in metadata"
            assert "depth_statistics" in metadata, "Missing depth_statistics in metadata"
            
            # Verify reasonable values
            assert metadata["valid_points"] > 0, "No valid points found"
            assert 0 <= metadata["point_confidence"] <= 1, f"Invalid point confidence: {metadata['point_confidence']}"
            assert 0 <= metadata["depth_quality"] <= 1, f"Invalid depth quality: {metadata['depth_quality']}"
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Point Cloud Processing",
                True,
                execution_time,
                {"metadata": metadata, "center_point": center_point, "depth_value": depth_value}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Point Cloud Processing",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_visualization_features(self) -> bool:
        """Test visualization enhancements"""
        start_time = time.time()
        
        try:
            if not self.processor:
                raise ValueError("Processor not initialized")
            
            # Test visualization configuration options
            viz_config = self.processor.DISPLAY_OPTIONS
            
            # Verify required configuration options exist
            required_options = [
                'show_coordinates', 'show_angles', 'coordinate_precision',
                'angle_precision', 'compact_display'
            ]
            
            for option in required_options:
                assert option in viz_config, f"Missing visualization option: {option}"
            
            # Test configuration updates
            original_compact = viz_config['compact_display']
            self.processor.DISPLAY_OPTIONS['compact_display'] = not original_compact
            
            # Verify configuration was updated
            assert self.processor.DISPLAY_OPTIONS['compact_display'] != original_compact, "Configuration update failed"
            
            # Restore original setting
            self.processor.DISPLAY_OPTIONS['compact_display'] = original_compact
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Visualization Enhancements",
                True,
                execution_time,
                {"config_options": len(required_options), "config_update": True}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Visualization Enhancements",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_performance_impact(self) -> bool:
        """Test performance impact of enhancements"""
        start_time = time.time()
        
        try:
            if not self.processor:
                raise ValueError("Processor not initialized")
            
            # Create test data
            test_depth = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)
            test_bbox = (200, 150, 400, 350)
            
            # Measure performance of key operations
            operations = []
            test_bbox_list = [200.0, 150.0, 400.0, 350.0]  # Convert to List[float]
            
            # Test weighted center point extraction
            op_start = time.perf_counter()
            last_result = None
            for _ in range(10):
                last_result = self.processor.calculate_weighted_center_point(test_depth, test_bbox_list)
            center_point_time = (time.perf_counter() - op_start) / 10
            operations.append(("center_point_extraction", center_point_time * 1000))  # Convert to ms
            
            # Test angle calculations
            if last_result and len(last_result) == 3:
                center_point, depth_val, metadata = last_result
                center_3d = self.processor.image_to_3d_coordinates(center_point, depth_val, self.processor.camera_intrinsics)
                
                op_start = time.perf_counter()
                for _ in range(100):
                    angles = self.processor.calculate_spherical_angles(*center_3d)
                angle_calc_time = (time.perf_counter() - op_start) / 100
                operations.append(("angle_calculation", angle_calc_time * 1000))
            
            # Test 3D coordinate conversion
            op_start = time.perf_counter()
            for _ in range(100):
                coords = self.processor.image_to_3d_coordinates((320, 240), 1000, self.processor.camera_intrinsics)
            coord_conv_time = (time.perf_counter() - op_start) / 100
            operations.append(("3d_conversion", coord_conv_time * 1000))
            
            # Verify performance meets requirements
            performance_requirements = {
                "center_point_extraction": 10.0,  # Max 10ms per extraction
                "angle_calculation": 0.1,         # Max 0.1ms per calculation
                "3d_conversion": 0.1               # Max 0.1ms per conversion
            }
            
            performance_results = {}
            for op_name, op_time in operations:
                performance_results[op_name] = op_time
                max_time = performance_requirements[op_name]
                assert op_time <= max_time, f"{op_name} too slow: {op_time:.3f}ms > {max_time}ms"
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Performance Impact",
                True,
                execution_time,
                {"performance_ms": performance_results, "requirements_met": True}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "Performance Impact",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def test_system_integration(self) -> bool:
        """Test complete system integration"""
        start_time = time.time()
        
        try:
            if not self.processor:
                raise ValueError("Processor not initialized")
            
            # Test complete detection pipeline with enhanced features
            # Create test image and depth data
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_depth = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)
            
            # Test weighted center point calculation and coordinate conversion
            test_bbox = [150.0, 100.0, 300.0, 250.0]  # x1, y1, x2, y2
            result = self.processor.calculate_weighted_center_point(test_depth, test_bbox)
            
            # Verify complete pipeline works
            assert result is not None, "Pipeline processing returned None"
            center_point, depth_value, metadata = result
            
            # Test 3D coordinate conversion
            center_3d = self.processor.image_to_3d_coordinates(center_point, depth_value, self.processor.camera_intrinsics)
            assert center_3d is not None, "3D coordinate conversion failed"
            
            # Test angle calculation
            azimuth, elevation = self.processor.calculate_spherical_angles(*center_3d)
            
            # Verify reasonable values from complete pipeline
            x, y = center_point
            assert 150 <= x <= 300, f"Center point X {x} outside bbox"
            assert 100 <= y <= 250, f"Center point Y {y} outside bbox"
            assert 500 <= depth_value <= 3000, f"Depth value {depth_value} outside range"
            assert -180 <= azimuth <= 180, f"Invalid azimuth: {azimuth}"
            assert -90 <= elevation <= 90, f"Invalid elevation: {elevation}"
            
            # Test configuration system integration
            config_update_test = {
                'point_cloud_processing': {'weighting_method': 'combined'},
                'angle_calculation': {'angle_units': 'degrees'},
                'visualization': {'show_coordinates': True}
            }
            self.processor.update_configuration(config_update_test)
            
            # Verify configuration updates were applied
            assert self.processor.POINT_CLOUD_CONFIG['weighting_method'] == 'combined'
            assert self.processor.ANGLE_CONFIG['angle_units'] == 'degrees'
            assert self.processor.DISPLAY_OPTIONS['show_coordinates'] == True
            
            integration_results = {
                "center_point": center_point,
                "depth_value": depth_value,
                "center_3d": center_3d,
                "azimuth": azimuth,
                "elevation": elevation,
                "metadata": metadata,
                "config_update": True
            }
            
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "System Integration",
                True,
                execution_time,
                {"integration_results": integration_results, "pipeline_complete": True}
            ))
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.test_results.append(TestResult(
                "System Integration",
                False,
                execution_time,
                {},
                str(e)
            ))
            return False
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("🧪 ENHANCED DETECTION TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results if result.passed)
        total_tests = len(self.test_results)
        total_time = sum(result.execution_time for result in self.test_results)
        
        print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
        print(f"⏱️  Total execution time: {total_time:.3f}s")
        print(f"✅ Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} | {result.test_name:<35} | {result.execution_time*1000:6.1f}ms")
            
            if not result.passed and result.error:
                print(f"    Error: {result.error}")
            elif result.details:
                key_detail = next(iter(result.details.items()))
                print(f"    {key_detail[0]}: {key_detail[1]}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Enhanced detection system is working correctly.")
        else:
            failed_tests = [r.test_name for r in self.test_results if not r.passed]
            print(f"\n⚠️  Failed tests: {', '.join(failed_tests)}")
        
        # Save detailed results
        self.save_test_results()
    
    def save_test_results(self):
        """Save test results to JSON file"""
        try:
            results_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total_tests": len(self.test_results),
                    "passed_tests": sum(1 for r in self.test_results if r.passed),
                    "total_time": sum(r.execution_time for r in self.test_results),
                    "success_rate": (sum(1 for r in self.test_results if r.passed) / len(self.test_results)) * 100
                },
                "detailed_results": [
                    {
                        "test_name": r.test_name,
                        "passed": r.passed,
                        "execution_time": r.execution_time,
                        "details": r.details,
                        "error": r.error
                    }
                    for r in self.test_results
                ]
            }
            
            with open("test_results_enhanced_detection.json", "w") as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"\n💾 Detailed test results saved to: test_results_enhanced_detection.json")
            
        except Exception as e:
            print(f"❌ Could not save test results: {e}")

def main():
    """Main test execution"""
    print("🚀 Starting Enhanced Object Detection Test Suite")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Create and run tester
    tester = EnhancedDetectionTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"\n🏁 Test suite completed with exit code: {exit_code}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()