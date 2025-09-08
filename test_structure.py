#!/usr/bin/env python3
"""
Simple structure validation script that doesn't require external dependencies.

This script tests that the package structure is correct and modules can be imported.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_package_structure():
    """Test that the package structure is correct."""
    print("Testing package structure...")
    
    expected_files = [
        'evaluation/__init__.py',
        'evaluation/metrics/__init__.py',
        'evaluation/metrics/base_metric.py',
        'evaluation/metrics/percentage_of_correct_keypoints.py',
        'evaluation/tests/__init__.py',
        'evaluation/tests/test_base_metric.py',
        'evaluation/tests/test_percentage_of_correct_keypoints.py',
        'setup.py',
        'requirements.txt',
        'pyproject.toml',
        'README.md'
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All expected files present")
        return True

def test_module_structure():
    """Test that modules have the expected structure without importing numpy."""
    print("\\nTesting module structure...")
    
    try:
        # Test base metric import
        from evaluation.metrics.base_metric import BaseMetric
        print("✓ BaseMetric imported successfully")
        
        # Check that BaseMetric is abstract
        if hasattr(BaseMetric, '__abstractmethods__'):
            print("✓ BaseMetric is properly abstract")
        else:
            print("✗ BaseMetric should be abstract")
            return False
        
        # Test that BaseMetric cannot be instantiated
        try:
            BaseMetric()
            print("✗ BaseMetric should not be instantiable")
            return False
        except TypeError:
            print("✓ BaseMetric correctly prevents direct instantiation")
        
        # Test PCK metric import (without instantiation)
        from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
        print("✓ PercentageOfCorrectKeyPoints imported successfully")
        
        # Test package-level import
        from evaluation import PercentageOfCorrectKeyPoints as PCK_from_package
        print("✓ Package-level import works")
        
        # Verify they are the same class
        if PCK_from_package is PercentageOfCorrectKeyPoints:
            print("✓ Package-level import points to correct class")
        else:
            print("✗ Package-level import inconsistency")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Module structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_interfaces():
    """Test class interfaces and method signatures."""
    print("\\nTesting class interfaces...")
    
    try:
        from evaluation.metrics.base_metric import BaseMetric
        from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
        
        # Check BaseMetric methods
        expected_base_methods = ['apply', '_compute_metric', '_validate_inputs', 'get_config']
        for method in expected_base_methods:
            if not hasattr(BaseMetric, method):
                print(f"✗ BaseMetric missing method: {method}")
                return False
        print("✓ BaseMetric has all expected methods")
        
        # Check PCK metric methods
        expected_pck_methods = ['apply', '_compute_metric', 'compute_detailed_results']
        for method in expected_pck_methods:
            if not hasattr(PercentageOfCorrectKeyPoints, method):
                print(f"✗ PercentageOfCorrectKeyPoints missing method: {method}")
                return False
        print("✓ PercentageOfCorrectKeyPoints has all expected methods")
        
        # Test inheritance
        if not issubclass(PercentageOfCorrectKeyPoints, BaseMetric):
            print("✗ PercentageOfCorrectKeyPoints should inherit from BaseMetric")
            return False
        print("✓ Inheritance structure is correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Class interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_docstrings():
    """Test that classes and methods have proper docstrings."""
    print("\\nTesting docstrings...")
    
    try:
        from evaluation.metrics.base_metric import BaseMetric
        from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
        
        # Check class docstrings
        if not BaseMetric.__doc__:
            print("✗ BaseMetric missing class docstring")
            return False
        print("✓ BaseMetric has class docstring")
        
        if not PercentageOfCorrectKeyPoints.__doc__:
            print("✗ PercentageOfCorrectKeyPoints missing class docstring")
            return False
        print("✓ PercentageOfCorrectKeyPoints has class docstring")
        
        # Check method docstrings
        if not BaseMetric.apply.__doc__:
            print("✗ BaseMetric.apply missing docstring")
            return False
        print("✓ BaseMetric.apply has docstring")
        
        if not PercentageOfCorrectKeyPoints._compute_metric.__doc__:
            print("✗ PercentageOfCorrectKeyPoints._compute_metric missing docstring")
            return False
        print("✓ PercentageOfCorrectKeyPoints._compute_metric has docstring")
        
        return True
        
    except Exception as e:
        print(f"✗ Docstring test failed: {e}")
        return False

def main():
    """Run all structure validation tests."""
    print("=" * 60)
    print("EVALUATION PACKAGE STRUCTURE VALIDATION")
    print("(Tests that don't require external dependencies)")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        test_package_structure,
        test_module_structure,
        test_class_interfaces,
        test_docstrings
    ]
    
    for test in tests:
        try:
            if not test():
                all_tests_passed = False
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            all_tests_passed = False
    
    print("\\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL STRUCTURE TESTS PASSED!")
        print("\\nThe package structure is correct and ready for use.")
        print("\\nTo test with actual data, install numpy and run:")
        print("  python3 test_implementation.py")
        print("  python3 example_usage.py")
    else:
        print("❌ SOME STRUCTURE TESTS FAILED!")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())