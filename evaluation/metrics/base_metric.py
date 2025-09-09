"""
Base Metric Module

Contains the abstract base class for all evaluation metrics.
This module implements the Template Method design pattern to ensure
consistent interface across all metric implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    
    This class defines the common interface that all metrics must implement.
    It follows the Template Method design pattern to ensure consistency
    and maintainability across different metric implementations.
    
    Attributes:
        name (str): The name of the metric
        config (Dict[str, Any]): Configuration parameters for the metric
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize the base metric.
        
        Args:
            name (Optional[str]): Name of the metric. If None, uses class name.
            **kwargs: Additional configuration parameters
        """
        self.name = name or self.__class__.__name__
        self.config = kwargs
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        This method can be overridden by subclasses to implement
        specific validation logic for their parameters.
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        pass
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Validate input arrays for shape and type compatibility.
        
        Args:
            y_true (np.ndarray): Ground truth heatmaps
            y_pred (np.ndarray): Predicted heatmaps
            
        Raises:
            TypeError: If inputs are not numpy arrays
            ValueError: If input shapes don't match or are invalid
        """
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise TypeError("Both y_true and y_pred must be numpy arrays")
        
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )
        
        if len(y_true.shape) != 4:
            raise ValueError(
                f"Expected 4D arrays with shape (batch_size, height, width, num_keypoints), "
                f"got shape {y_true.shape}"
            )
        
        if y_true.shape[0] == 0:
            raise ValueError("Batch size cannot be zero")
    
    @abstractmethod
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute the actual metric value.
        
        This method must be implemented by all subclasses.
        
        Args:
            y_true (np.ndarray): Ground truth heatmaps of shape 
                                (batch_size, height, width, num_keypoints)
            y_pred (np.ndarray): Predicted heatmaps of shape 
                                (batch_size, height, width, num_keypoints)
        
        Returns:
            Union[float, np.ndarray]: The computed metric value(s)
        """
        pass
    
    def apply(self, y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
        """
        Apply the metric to the given predictions and ground truth.
        
        This is the main public interface for computing metrics.
        It implements the Template Method pattern by calling the
        validation methods and then the abstract _compute_metric method.
        
        Args:
            y_true (np.ndarray): Ground truth heatmaps of shape 
                                (batch_size, height, width, num_keypoints)
            y_pred (np.ndarray): Predicted heatmaps of shape 
                                (batch_size, height, width, num_keypoints)
        
        Returns:
            Union[float, np.ndarray]: The computed metric value(s)
            
        Raises:
            TypeError: If inputs are not numpy arrays
            ValueError: If input shapes are incompatible
        """
        self._validate_inputs(y_true, y_pred)
        return self._compute_metric(y_true, y_pred)
    
    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name}(config={self.config})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the metric."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters of the metric.
        
        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters
        """
        return self.config.copy()