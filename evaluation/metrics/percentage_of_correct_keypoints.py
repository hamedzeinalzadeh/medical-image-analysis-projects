"""
Percentage of Correct Keypoints (PCK) Metric

This module implements the PCK metric for evaluating keypoint detection models.
PCK measures the percentage of predicted keypoints that fall within a specified
distance threshold from the ground truth keypoints.
"""

from typing import Optional, Union
import numpy as np
from evaluation.metrics.base_metric import BaseMetric


class PercentageOfCorrectKeyPoints(BaseMetric):
    """
    Percentage of Correct Keypoints (PCK) metric.
    
    This metric calculates the percentage of predicted keypoints that are within
    a specified distance threshold from the ground truth keypoints. The distance
    threshold is defined as a fraction of the image height.
    
    The metric works by:
    1. Finding the coordinates of maximum values in both true and predicted heatmaps
    2. Computing the Euclidean distance between corresponding keypoints
    3. Counting keypoints where the distance is below the threshold
    4. Returning the percentage of correct keypoints
    
    Attributes:
        relative_distance_threshold (float): The distance threshold as a fraction 
                                           of image height (default: 0.1)
        return_per_keypoint (bool): Whether to return results per keypoint or 
                                  averaged across all keypoints (default: False)
    """
    
    def __init__(
        self, 
        relative_distance_threshold: float = 0.1,
        return_per_keypoint: bool = False,
        name: Optional[str] = None
    ) -> None:
        """
        Initialize the PCK metric.
        
        Args:
            relative_distance_threshold (float): The distance threshold as a fraction
                                               of the image height. Must be positive.
            return_per_keypoint (bool): If True, returns PCK for each keypoint separately.
                                      If False, returns the average PCK across all keypoints.
            name (Optional[str]): Custom name for the metric
            
        Raises:
            ValueError: If relative_distance_threshold is not positive
        """
        super().__init__(
            name=name,
            relative_distance_threshold=relative_distance_threshold,
            return_per_keypoint=return_per_keypoint
        )
        self.relative_distance_threshold = relative_distance_threshold
        self.return_per_keypoint = return_per_keypoint
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If relative_distance_threshold is not positive
        """
        if self.relative_distance_threshold <= 0:
            raise ValueError(
                f"relative_distance_threshold must be positive, "
                f"got {self.relative_distance_threshold}"
            )
    
    def _extract_keypoint_coordinates(self, heatmaps: np.ndarray) -> np.ndarray:
        """
        Extract keypoint coordinates from heatmaps by finding maximum values.
        
        Args:
            heatmaps (np.ndarray): Heatmaps of shape (batch_size, height, width, num_keypoints)
            
        Returns:
            np.ndarray: Coordinates of shape (batch_size, num_keypoints, 2) where
                       the last dimension contains (row, col) coordinates
        """
        batch_size, height, width, num_keypoints = heatmaps.shape
        coordinates = np.zeros((batch_size, num_keypoints, 2), dtype=np.float32)
        
        for batch_idx in range(batch_size):
            for kp_idx in range(num_keypoints):
                # Find the coordinates of the maximum value in the heatmap
                heatmap = heatmaps[batch_idx, :, :, kp_idx]
                max_idx = np.unravel_index(np.argmax(heatmap), (height, width))
                coordinates[batch_idx, kp_idx] = [max_idx[0], max_idx[1]]  # (row, col)
        
        return coordinates
    
    def _compute_distances(self, true_coords: np.ndarray, pred_coords: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between true and predicted keypoint coordinates.
        
        Args:
            true_coords (np.ndarray): True coordinates of shape (batch_size, num_keypoints, 2)
            pred_coords (np.ndarray): Predicted coordinates of shape (batch_size, num_keypoints, 2)
            
        Returns:
            np.ndarray: Distances of shape (batch_size, num_keypoints)
        """
        # Compute Euclidean distance for each keypoint
        distances = np.linalg.norm(true_coords - pred_coords, axis=2)
        return distances
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute the PCK metric.
        
        Args:
            y_true (np.ndarray): Ground truth heatmaps of shape 
                                (batch_size, height, width, num_keypoints)
            y_pred (np.ndarray): Predicted heatmaps of shape 
                                (batch_size, height, width, num_keypoints)
        
        Returns:
            Union[float, np.ndarray]: PCK value(s). If return_per_keypoint is True,
                                    returns array of shape (num_keypoints,).
                                    Otherwise, returns a single float value.
        """
        batch_size, height, width, num_keypoints = y_true.shape
        
        # Calculate the distance threshold in pixels
        distance_threshold = self.relative_distance_threshold * height
        
        # Extract keypoint coordinates from heatmaps
        true_coords = self._extract_keypoint_coordinates(y_true)
        pred_coords = self._extract_keypoint_coordinates(y_pred)
        
        # Compute distances between true and predicted keypoints
        distances = self._compute_distances(true_coords, pred_coords)
        
        # Determine which keypoints are correct (within threshold)
        correct_keypoints = distances <= distance_threshold
        
        if self.return_per_keypoint:
            # Return PCK for each keypoint separately
            pck_per_keypoint = np.mean(correct_keypoints, axis=0)
            return pck_per_keypoint
        else:
            # Return average PCK across all keypoints
            total_correct = np.sum(correct_keypoints)
            total_keypoints = batch_size * num_keypoints
            pck = total_correct / total_keypoints
            return float(pck)
    
    def compute_detailed_results(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute detailed PCK results including per-keypoint and per-sample statistics.
        
        Args:
            y_true (np.ndarray): Ground truth heatmaps
            y_pred (np.ndarray): Predicted heatmaps
            
        Returns:
            dict: Dictionary containing:
                - 'overall_pck': Overall PCK score
                - 'per_keypoint_pck': PCK for each keypoint
                - 'per_sample_pck': PCK for each sample in the batch
                - 'distance_threshold_pixels': The distance threshold used in pixels
                - 'total_keypoints': Total number of keypoints evaluated
                - 'correct_keypoints': Number of correct keypoints
        """
        self._validate_inputs(y_true, y_pred)
        
        batch_size, height, width, num_keypoints = y_true.shape
        distance_threshold = self.relative_distance_threshold * height
        
        # Extract coordinates and compute distances
        true_coords = self._extract_keypoint_coordinates(y_true)
        pred_coords = self._extract_keypoint_coordinates(y_pred)
        distances = self._compute_distances(true_coords, pred_coords)
        
        # Compute correctness matrix
        correct_keypoints = distances <= distance_threshold
        
        # Compute various statistics
        overall_pck = np.mean(correct_keypoints)
        per_keypoint_pck = np.mean(correct_keypoints, axis=0)
        per_sample_pck = np.mean(correct_keypoints, axis=1)
        
        return {
            'overall_pck': float(overall_pck),
            'per_keypoint_pck': per_keypoint_pck.tolist(),
            'per_sample_pck': per_sample_pck.tolist(),
            'distance_threshold_pixels': float(distance_threshold),
            'total_keypoints': int(batch_size * num_keypoints),
            'correct_keypoints': int(np.sum(correct_keypoints)),
            'distances': distances.tolist(),
            'true_coordinates': true_coords.tolist(),
            'predicted_coordinates': pred_coords.tolist()
        }