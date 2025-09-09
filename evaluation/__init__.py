"""
Evaluation Package

A Python package for evaluating keypoint detection models.
Provides metrics and utilities for assessing the performance of keypoint detection algorithms.
"""

__version__ = "0.1.0"
__author__ = "Evaluation Package Team"

# Import main metric classes for easy access
from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints

__all__ = [
    "PercentageOfCorrectKeyPoints",
]