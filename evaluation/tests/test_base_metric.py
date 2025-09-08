"""
Unit tests for BaseMetric abstract class.

This module contains tests for the base metric functionality,
including validation and abstract method enforcement.
"""

import pytest
import numpy as np
from evaluation.metrics.base_metric import BaseMetric


class ConcreteMetric(BaseMetric):
    """Concrete implementation of BaseMetric for testing purposes."""
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Simple test implementation that returns the mean difference."""
        return float(np.mean(np.abs(y_true - y_pred)))


class TestBaseMetric:
    """Test suite for BaseMetric abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric()
    
    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be instantiated and used."""
        metric = ConcreteMetric()
        assert isinstance(metric, BaseMetric)
        assert metric.name == "ConcreteMetric"
    
    def test_initialization_with_custom_name(self):
        """Test initialization with custom name."""
        metric = ConcreteMetric(name="CustomName")
        assert metric.name == "CustomName"
    
    def test_initialization_with_config(self):
        """Test initialization with configuration parameters."""
        metric = ConcreteMetric(param1=10, param2="test")
        assert metric.config["param1"] == 10
        assert metric.config["param2"] == "test"
    
    def test_input_validation_success(self):
        """Test successful input validation."""
        metric = ConcreteMetric()
        y_true = np.random.rand(2, 100, 100, 3)
        y_pred = np.random.rand(2, 100, 100, 3)
        
        # Should not raise any exception
        result = metric.apply(y_true, y_pred)
        assert isinstance(result, float)
    
    def test_input_validation_type_error(self):
        """Test input validation with wrong types."""
        metric = ConcreteMetric()
        
        with pytest.raises(TypeError, match="must be numpy arrays"):
            metric.apply([1, 2, 3], np.zeros((2, 100, 100, 3)))
        
        with pytest.raises(TypeError, match="must be numpy arrays"):
            metric.apply(np.zeros((2, 100, 100, 3)), [1, 2, 3])
    
    def test_input_validation_shape_mismatch(self):
        """Test input validation with shape mismatches."""
        metric = ConcreteMetric()
        
        y_true = np.zeros((2, 100, 100, 3))
        y_pred = np.zeros((3, 100, 100, 3))  # Different batch size
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.apply(y_true, y_pred)
    
    def test_input_validation_wrong_dimensions(self):
        """Test input validation with wrong number of dimensions."""
        metric = ConcreteMetric()
        
        y_true = np.zeros((100, 100, 3))  # 3D instead of 4D
        y_pred = np.zeros((100, 100, 3))
        
        with pytest.raises(ValueError, match="Expected 4D arrays"):
            metric.apply(y_true, y_pred)
    
    def test_input_validation_zero_batch_size(self):
        """Test input validation with zero batch size."""
        metric = ConcreteMetric()
        
        y_true = np.zeros((0, 100, 100, 3))
        y_pred = np.zeros((0, 100, 100, 3))
        
        with pytest.raises(ValueError, match="Batch size cannot be zero"):
            metric.apply(y_true, y_pred)
    
    def test_string_representation(self):
        """Test string representation methods."""
        metric = ConcreteMetric(name="TestMetric", param1=5)
        
        str_repr = str(metric)
        assert "TestMetric" in str_repr
        assert "param1" in str_repr
        
        repr_str = repr(metric)
        assert "ConcreteMetric" in repr_str
        assert "TestMetric" in repr_str
    
    def test_get_config(self):
        """Test get_config method."""
        metric = ConcreteMetric(param1=10, param2="test")
        config = metric.get_config()
        
        assert config["param1"] == 10
        assert config["param2"] == "test"
        
        # Test that it returns a copy
        config["param1"] = 20
        assert metric.config["param1"] == 10  # Original unchanged
    
    def test_validate_config_override(self):
        """Test that _validate_config can be overridden."""
        
        class ValidatingMetric(BaseMetric):
            def _validate_config(self):
                if self.config.get("threshold", 0) <= 0:
                    raise ValueError("threshold must be positive")
            
            def _compute_metric(self, y_true, y_pred):
                return 0.0
        
        # Should work with valid config
        metric = ValidatingMetric(threshold=0.1)
        assert metric.config["threshold"] == 0.1
        
        # Should raise error with invalid config
        with pytest.raises(ValueError, match="threshold must be positive"):
            ValidatingMetric(threshold=0)


if __name__ == "__main__":
    pytest.main([__file__])