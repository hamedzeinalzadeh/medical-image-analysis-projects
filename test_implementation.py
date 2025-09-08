#!/usr/bin/env python3
"""
Simple validation script to test the evaluation package implementation.

This script tests the basic functionality without requiring external dependencies
beyond numpy (which should be available in most environments).
"""

import sys
import os
import traceback

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from evaluation.metrics.base_metric import BaseMetric
        print("✓ BaseMetric imported successfully")
    except ImportError as e:
        print(f"✗ BaseMetric import failed: {e}")
        return False
    
    try:
        from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
        print("✓ PercentageOfCorrectKeyPoints imported successfully")
    except ImportError as e:
        print(f"✗ PercentageOfCorrectKeyPoints import failed: {e}")
        return False
    
    try:
        from evaluation import PercentageOfCorrectKeyPoints as PCK
        print("✓ Package-level import successful")
    except ImportError as e:
        print(f"✗ Package-level import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the PCK metric."""
    print("\\nTesting basic functionality...")
    
    try:
        import numpy as np
        from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
        
        # Test 1: Perfect predictions
        print("Test 1: Perfect predictions")
        y_true = np.zeros((1, 100, 100, 1))
        y_pred = np.zeros((1, 100, 100, 1))
        y_true[0, 50, 50, 0] = 1.0
        y_pred[0, 50, 50, 0] = 1.0
        
        metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)
        result = metric.apply(y_true, y_pred)
        
        if result == 1.0:
            print("✓ Perfect predictions test passed")
        else:
            print(f"✗ Perfect predictions test failed: expected 1.0, got {result}")
            return False
        
        # Test 2: Completely wrong predictions
        print("Test 2: Completely wrong predictions")
        y_pred[0, 50, 50, 0] = 0.0  # Remove correct prediction
        y_pred[0, 10, 10, 0] = 1.0  # Place far away
        
        result = metric.apply(y_true, y_pred)
        
        if result == 0.0:
            print("✓ Wrong predictions test passed")
        else:
            print(f"✗ Wrong predictions test failed: expected 0.0, got {result}")
            return False
        
        # Test 3: Predictions within threshold
        print("Test 3: Predictions within threshold")
        y_pred[0, 10, 10, 0] = 0.0  # Remove wrong prediction
        y_pred[0, 55, 55, 0] = 1.0  # Place within threshold (distance = ~7 pixels)
        
        result = metric.apply(y_true, y_pred)
        
        if result == 1.0:
            print("✓ Within threshold test passed")
        else:
            print(f"✗ Within threshold test failed: expected 1.0, got {result}")
            return False
        
        # Test 4: Multiple keypoints
        print("Test 4: Multiple keypoints")
        y_true_multi = np.zeros((2, 100, 100, 3))
        y_pred_multi = np.zeros((2, 100, 100, 3))
        
        # Set up some keypoints
        positions = [
            (0, 0, 25, 25), (0, 1, 50, 50), (0, 2, 75, 75),
            (1, 0, 30, 30), (1, 1, 60, 60), (1, 2, 80, 80)
        ]
        
        for batch, kp, row, col in positions:
            y_true_multi[batch, row, col, kp] = 1.0
            y_pred_multi[batch, row, col, kp] = 1.0  # Perfect predictions
        
        result = metric.apply(y_true_multi, y_pred_multi)
        
        if result == 1.0:
            print("✓ Multiple keypoints test passed")
        else:
            print(f"✗ Multiple keypoints test failed: expected 1.0, got {result}")
            return False
        
        # Test 5: Per-keypoint results
        print("Test 5: Per-keypoint results")
        metric_per_kp = PercentageOfCorrectKeyPoints(
            relative_distance_threshold=0.1, 
            return_per_keypoint=True
        )
        
        # Make only first keypoint correct
        y_pred_multi[0, 50, 50, 1] = 0.0  # Remove correct prediction
        y_pred_multi[0, 10, 10, 1] = 1.0  # Place incorrectly
        y_pred_multi[1, 60, 60, 1] = 0.0  # Remove correct prediction
        y_pred_multi[1, 10, 10, 1] = 1.0  # Place incorrectly
        
        result = metric_per_kp.apply(y_true_multi, y_pred_multi)
        
        if (isinstance(result, np.ndarray) and len(result) == 3 and 
            result[0] == 1.0 and result[1] == 0.0):
            print("✓ Per-keypoint results test passed")
        else:
            print(f"✗ Per-keypoint results test failed: {result}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed with exception: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and validation."""
    print("\\nTesting error handling...")
    
    try:
        import numpy as np
        from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
        
        metric = PercentageOfCorrectKeyPoints()
        
        # Test 1: Invalid threshold
        print("Test 1: Invalid threshold validation")
        try:
            PercentageOfCorrectKeyPoints(relative_distance_threshold=0)
            print("✗ Should have raised ValueError for zero threshold")
            return False
        except ValueError:
            print("✓ Correctly rejected zero threshold")
        
        # Test 2: Shape mismatch
        print("Test 2: Shape mismatch validation")
        y_true = np.zeros((2, 100, 100, 3))
        y_pred = np.zeros((2, 100, 100, 2))  # Different number of keypoints
        
        try:
            metric.apply(y_true, y_pred)
            print("✗ Should have raised ValueError for shape mismatch")
            return False
        except ValueError:
            print("✓ Correctly detected shape mismatch")
        
        # Test 3: Wrong dimensions
        print("Test 3: Wrong dimensions validation")
        y_true = np.zeros((100, 100, 3))  # 3D instead of 4D
        y_pred = np.zeros((100, 100, 3))
        
        try:
            metric.apply(y_true, y_pred)
            print("✗ Should have raised ValueError for wrong dimensions")
            return False
        except ValueError:
            print("✓ Correctly detected wrong dimensions")
        
        # Test 4: Non-numpy arrays
        print("Test 4: Non-numpy array validation")
        y_true = [[1, 2], [3, 4]]  # List instead of numpy array
        y_pred = np.zeros((2, 100, 100, 3))
        
        try:
            metric.apply(y_true, y_pred)
            print("✗ Should have raised TypeError for non-numpy array")
            return False
        except TypeError:
            print("✓ Correctly detected non-numpy array")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed with exception: {e}")
        traceback.print_exc()
        return False

def test_detailed_results():
    """Test detailed results functionality."""
    print("\\nTesting detailed results...")
    
    try:
        import numpy as np
        from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
        
        metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)
        
        # Create simple test case
        y_true = np.zeros((1, 100, 100, 2))
        y_pred = np.zeros((1, 100, 100, 2))
        
        y_true[0, 50, 50, 0] = 1.0  # First keypoint
        y_true[0, 60, 60, 1] = 1.0  # Second keypoint
        
        y_pred[0, 50, 50, 0] = 1.0  # First keypoint correct
        y_pred[0, 90, 90, 1] = 1.0  # Second keypoint incorrect
        
        results = metric.compute_detailed_results(y_true, y_pred)
        
        # Check that all expected keys are present
        expected_keys = [
            'overall_pck', 'per_keypoint_pck', 'per_sample_pck',
            'distance_threshold_pixels', 'total_keypoints', 'correct_keypoints',
            'distances', 'true_coordinates', 'predicted_coordinates'
        ]
        
        for key in expected_keys:
            if key not in results:
                print(f"✗ Missing key in detailed results: {key}")
                return False
        
        # Check values
        if results['overall_pck'] != 0.5:  # 1 correct out of 2
            print(f"✗ Wrong overall PCK: expected 0.5, got {results['overall_pck']}")
            return False
        
        if results['total_keypoints'] != 2:
            print(f"✗ Wrong total keypoints: expected 2, got {results['total_keypoints']}")
            return False
        
        if results['correct_keypoints'] != 1:
            print(f"✗ Wrong correct keypoints: expected 1, got {results['correct_keypoints']}")
            return False
        
        print("✓ Detailed results test passed")
        return True
        
    except Exception as e:
        print(f"✗ Detailed results test failed with exception: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("EVALUATION PACKAGE VALIDATION")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        test_imports,
        test_basic_functionality,
        test_error_handling,
        test_detailed_results
    ]
    
    for test in tests:
        try:
            if not test():
                all_tests_passed = False
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()
            all_tests_passed = False
    
    print("\\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! The evaluation package is working correctly.")
        print("\\nYou can now use the package as described in the README:")
        print("\\n  from evaluation import PercentageOfCorrectKeyPoints")
        print("  metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)")
        print("  result = metric.apply(y_true, y_pred)")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())