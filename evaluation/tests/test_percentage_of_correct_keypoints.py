"""
Unit tests for PercentageOfCorrectKeyPoints metric.

This module contains comprehensive tests for the PCK metric implementation,
including edge cases, error handling, and various scenarios.
"""

import pytest
import numpy as np
from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
from evaluation.metrics.base_metric import BaseMetric


class TestPercentageOfCorrectKeyPoints:
    """Test suite for PercentageOfCorrectKeyPoints metric."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)
        self.batch_size = 2
        self.height = 100
        self.width = 100
        self.num_keypoints = 3
    
    def create_test_heatmaps(self, true_positions, pred_positions):
        """
        Create test heatmaps with keypoints at specified positions.
        
        Args:
            true_positions: List of (batch, keypoint, row, col) tuples for true keypoints
            pred_positions: List of (batch, keypoint, row, col) tuples for predicted keypoints
            
        Returns:
            Tuple of (y_true, y_pred) heatmap arrays
        """
        y_true = np.zeros((self.batch_size, self.height, self.width, self.num_keypoints))
        y_pred = np.zeros((self.batch_size, self.height, self.width, self.num_keypoints))
        
        for batch, kp, row, col in true_positions:
            y_true[batch, row, col, kp] = 1.0
            
        for batch, kp, row, col in pred_positions:
            y_pred[batch, row, col, kp] = 1.0
            
        return y_true, y_pred
    
    def test_inheritance(self):
        """Test that PercentageOfCorrectKeyPoints inherits from BaseMetric."""
        assert isinstance(self.metric, BaseMetric)
        assert hasattr(self.metric, 'apply')
        assert hasattr(self.metric, '_compute_metric')
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        metric = PercentageOfCorrectKeyPoints()
        assert metric.relative_distance_threshold == 0.1
        assert metric.return_per_keypoint == False
        assert metric.name == "PercentageOfCorrectKeyPoints"
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        metric = PercentageOfCorrectKeyPoints(
            relative_distance_threshold=0.2,
            return_per_keypoint=True,
            name="CustomPCK"
        )
        assert metric.relative_distance_threshold == 0.2
        assert metric.return_per_keypoint == True
        assert metric.name == "CustomPCK"
    
    def test_initialization_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="relative_distance_threshold must be positive"):
            PercentageOfCorrectKeyPoints(relative_distance_threshold=0)
        
        with pytest.raises(ValueError, match="relative_distance_threshold must be positive"):
            PercentageOfCorrectKeyPoints(relative_distance_threshold=-0.1)
    
    def test_perfect_predictions(self):
        """Test PCK with perfect predictions (all keypoints match exactly)."""
        # Create heatmaps where predicted and true keypoints are identical
        true_positions = [(0, 0, 50, 50), (0, 1, 60, 40), (0, 2, 30, 70),
                         (1, 0, 25, 25), (1, 1, 75, 75), (1, 2, 45, 55)]
        pred_positions = true_positions.copy()
        
        y_true, y_pred = self.create_test_heatmaps(true_positions, pred_positions)
        result = self.metric.apply(y_true, y_pred)
        
        assert result == 1.0, f"Expected PCK=1.0 for perfect predictions, got {result}"
    
    def test_all_incorrect_predictions(self):
        """Test PCK with all predictions outside threshold."""
        # Place true keypoints at one corner and predicted at opposite corner
        true_positions = [(0, 0, 10, 10), (0, 1, 10, 10), (0, 2, 10, 10),
                         (1, 0, 10, 10), (1, 1, 10, 10), (1, 2, 10, 10)]
        pred_positions = [(0, 0, 90, 90), (0, 1, 90, 90), (0, 2, 90, 90),
                         (1, 0, 90, 90), (1, 1, 90, 90), (1, 2, 90, 90)]
        
        y_true, y_pred = self.create_test_heatmaps(true_positions, pred_positions)
        result = self.metric.apply(y_true, y_pred)
        
        assert result == 0.0, f"Expected PCK=0.0 for all incorrect predictions, got {result}"
    
    def test_partial_correct_predictions(self):
        """Test PCK with some correct and some incorrect predictions."""
        # Mix of correct and incorrect predictions
        true_positions = [(0, 0, 50, 50), (0, 1, 60, 40), (0, 2, 30, 70),
                         (1, 0, 25, 25), (1, 1, 75, 75), (1, 2, 45, 55)]
        
        # First 3 keypoints correct (within threshold), last 3 incorrect
        pred_positions = [(0, 0, 50, 50), (0, 1, 60, 40), (0, 2, 30, 70),  # Correct
                         (1, 0, 90, 90), (1, 1, 10, 10), (1, 2, 90, 10)]   # Incorrect
        
        y_true, y_pred = self.create_test_heatmaps(true_positions, pred_positions)
        result = self.metric.apply(y_true, y_pred)
        
        expected = 3.0 / 6.0  # 3 correct out of 6 total keypoints
        assert abs(result - expected) < 1e-6, f"Expected PCK={expected}, got {result}"
    
    def test_threshold_boundary(self):
        """Test PCK at the threshold boundary."""
        # Create predictions exactly at the threshold distance
        threshold_distance = 0.1 * self.height  # 10 pixels for 100px height
        
        true_positions = [(0, 0, 50, 50)]
        # Place predicted keypoint exactly at threshold distance
        pred_positions = [(0, 0, 50 + int(threshold_distance), 50)]
        
        y_true, y_pred = self.create_test_heatmaps(true_positions, pred_positions)
        
        # Pad with zeros for other keypoints and batch
        y_true = y_true[:1, :, :, :1]  # Use only first batch and first keypoint
        y_pred = y_pred[:1, :, :, :1]
        
        result = self.metric.apply(y_true, y_pred)
        assert result == 1.0, f"Expected keypoint at threshold distance to be correct"
        
        # Test just outside threshold
        pred_positions = [(0, 0, 50 + int(threshold_distance) + 1, 50)]
        y_true, y_pred = self.create_test_heatmaps(true_positions, pred_positions)
        y_true = y_true[:1, :, :, :1]
        y_pred = y_pred[:1, :, :, :1]
        
        result = self.metric.apply(y_true, y_pred)
        assert result == 0.0, f"Expected keypoint outside threshold to be incorrect"
    
    def test_return_per_keypoint(self):
        """Test PCK with return_per_keypoint=True."""
        metric = PercentageOfCorrectKeyPoints(
            relative_distance_threshold=0.1, 
            return_per_keypoint=True
        )
        
        # Create scenario where keypoint 0 is always correct, others are incorrect
        true_positions = [(0, 0, 50, 50), (0, 1, 60, 40), (0, 2, 30, 70),
                         (1, 0, 25, 25), (1, 1, 75, 75), (1, 2, 45, 55)]
        pred_positions = [(0, 0, 50, 50), (0, 1, 90, 90), (0, 2, 90, 90),  # kp0 correct, others wrong
                         (1, 0, 25, 25), (1, 1, 10, 10), (1, 2, 10, 10)]   # kp0 correct, others wrong
        
        y_true, y_pred = self.create_test_heatmaps(true_positions, pred_positions)
        result = metric.apply(y_true, y_pred)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == self.num_keypoints
        assert result[0] == 1.0  # Keypoint 0 always correct
        assert result[1] == 0.0  # Keypoint 1 always incorrect
        assert result[2] == 0.0  # Keypoint 2 always incorrect
    
    def test_input_validation_shape_mismatch(self):
        """Test input validation for shape mismatches."""
        y_true = np.zeros((2, 100, 100, 3))
        y_pred = np.zeros((2, 100, 100, 2))  # Different number of keypoints
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            self.metric.apply(y_true, y_pred)
    
    def test_input_validation_wrong_dimensions(self):
        """Test input validation for wrong number of dimensions."""
        y_true = np.zeros((100, 100, 3))  # 3D instead of 4D
        y_pred = np.zeros((100, 100, 3))
        
        with pytest.raises(ValueError, match="Expected 4D arrays"):
            self.metric.apply(y_true, y_pred)
    
    def test_input_validation_non_numpy_arrays(self):
        """Test input validation for non-numpy arrays."""
        y_true = [[1, 2], [3, 4]]  # List instead of numpy array
        y_pred = np.zeros((2, 100, 100, 3))
        
        with pytest.raises(TypeError, match="must be numpy arrays"):
            self.metric.apply(y_true, y_pred)
    
    def test_input_validation_zero_batch_size(self):
        """Test input validation for zero batch size."""
        y_true = np.zeros((0, 100, 100, 3))
        y_pred = np.zeros((0, 100, 100, 3))
        
        with pytest.raises(ValueError, match="Batch size cannot be zero"):
            self.metric.apply(y_true, y_pred)
    
    def test_different_threshold_values(self):
        """Test PCK with different threshold values."""
        # Create a scenario with known distances
        true_positions = [(0, 0, 50, 50)]
        pred_positions = [(0, 0, 55, 50)]  # 5 pixels away
        
        y_true, y_pred = self.create_test_heatmaps(true_positions, pred_positions)
        y_true = y_true[:1, :, :, :1]
        y_pred = y_pred[:1, :, :, :1]
        
        # With threshold 0.1 (10 pixels), should be correct
        metric1 = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)
        result1 = metric1.apply(y_true, y_pred)
        assert result1 == 1.0
        
        # With threshold 0.04 (4 pixels), should be incorrect
        metric2 = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.04)
        result2 = metric2.apply(y_true, y_pred)
        assert result2 == 0.0
    
    def test_compute_detailed_results(self):
        """Test the compute_detailed_results method."""
        true_positions = [(0, 0, 50, 50), (0, 1, 60, 40)]
        pred_positions = [(0, 0, 50, 50), (0, 1, 90, 90)]  # First correct, second incorrect
        
        y_true, y_pred = self.create_test_heatmaps(true_positions, pred_positions)
        y_true = y_true[:1, :, :, :2]  # Use only first batch and first 2 keypoints
        y_pred = y_pred[:1, :, :, :2]
        
        results = self.metric.compute_detailed_results(y_true, y_pred)
        
        assert 'overall_pck' in results
        assert 'per_keypoint_pck' in results
        assert 'per_sample_pck' in results
        assert 'distance_threshold_pixels' in results
        assert 'total_keypoints' in results
        assert 'correct_keypoints' in results
        
        assert results['overall_pck'] == 0.5  # 1 correct out of 2
        assert len(results['per_keypoint_pck']) == 2
        assert results['per_keypoint_pck'][0] == 1.0  # First keypoint correct
        assert results['per_keypoint_pck'][1] == 0.0  # Second keypoint incorrect
        assert results['distance_threshold_pixels'] == 10.0  # 0.1 * 100
        assert results['total_keypoints'] == 2
        assert results['correct_keypoints'] == 1
    
    def test_string_representations(self):
        """Test string representations of the metric."""
        metric = PercentageOfCorrectKeyPoints(
            relative_distance_threshold=0.2, 
            name="TestPCK"
        )
        
        str_repr = str(metric)
        assert "TestPCK" in str_repr
        assert "0.2" in str_repr
        
        repr_str = repr(metric)
        assert "PercentageOfCorrectKeyPoints" in repr_str
        assert "TestPCK" in repr_str
    
    def test_get_config(self):
        """Test the get_config method."""
        metric = PercentageOfCorrectKeyPoints(
            relative_distance_threshold=0.15,
            return_per_keypoint=True
        )
        
        config = metric.get_config()
        assert config['relative_distance_threshold'] == 0.15
        assert config['return_per_keypoint'] == True
        
        # Ensure it's a copy (modifications don't affect original)
        config['relative_distance_threshold'] = 0.5
        assert metric.relative_distance_threshold == 0.15
    
    def test_edge_case_single_pixel_image(self):
        """Test with minimal image size."""
        y_true = np.zeros((1, 1, 1, 1))
        y_pred = np.zeros((1, 1, 1, 1))
        y_true[0, 0, 0, 0] = 1.0
        y_pred[0, 0, 0, 0] = 1.0
        
        result = self.metric.apply(y_true, y_pred)
        assert result == 1.0
    
    def test_multiple_maxima_in_heatmap(self):
        """Test behavior when heatmap has multiple maximum values."""
        y_true = np.zeros((1, 10, 10, 1))
        y_pred = np.zeros((1, 10, 10, 1))
        
        # Create multiple maxima in true heatmap
        y_true[0, 5, 5, 0] = 1.0
        y_true[0, 6, 6, 0] = 1.0  # Another maximum
        
        # Single maximum in predicted heatmap
        y_pred[0, 5, 5, 0] = 1.0
        
        # Should still work (numpy.argmax returns first occurrence)
        result = self.metric.apply(y_true, y_pred)
        assert isinstance(result, float)
        assert 0 <= result <= 1


if __name__ == "__main__":
    pytest.main([__file__])