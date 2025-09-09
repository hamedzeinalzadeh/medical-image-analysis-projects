#!/usr/bin/env python3
"""
Example usage of the evaluation package.

This script demonstrates how to use the PercentageOfCorrectKeyPoints metric
exactly as specified in the task requirements.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints

def main():
    """Demonstrate the usage as specified in the task."""
    
    print("Evaluation Package - Example Usage")
    print("=" * 50)
    
    # Create sample data as specified in the task
    print("Creating sample data...")
    batch_size, h, w, num_keypoints = 4, 256, 256, 17
    
    # Create random heatmaps (in practice, these would come from your model)
    y_true = np.random.rand(batch_size, h, w, num_keypoints)
    y_pred = np.random.rand(batch_size, h, w, num_keypoints)
    
    # For demonstration, let's create some realistic keypoints
    # by setting specific locations to have high values
    np.random.seed(42)  # For reproducible results
    
    for batch in range(batch_size):
        for kp in range(num_keypoints):
            # Random ground truth keypoint location
            true_y, true_x = np.random.randint(20, h-20, 2)
            y_true[batch, :, :, kp] *= 0.1  # Reduce background
            y_true[batch, true_y, true_x, kp] = 1.0  # Set keypoint
            
            # Predicted keypoint with some noise
            pred_y = true_y + np.random.randint(-15, 16)
            pred_x = true_x + np.random.randint(-15, 16)
            pred_y = max(0, min(h-1, pred_y))
            pred_x = max(0, min(w-1, pred_x))
            
            y_pred[batch, :, :, kp] *= 0.1  # Reduce background
            y_pred[batch, pred_y, pred_x, kp] = 1.0  # Set predicted keypoint
    
    print(f"Data shape: {y_true.shape}")
    print(f"Value range - True: [{y_true.min():.3f}, {y_true.max():.3f}]")
    print(f"Value range - Pred: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    
    # Use the metric exactly as specified in the task
    print("\\nUsing the metric as specified in the task:")
    print("metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)")
    print("result = metric.apply(y_true, y_pred)")
    
    metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)
    result = metric.apply(y_true, y_pred)
    
    print(f"\\nPCK Result: {result:.4f}")
    
    # Demonstrate additional features
    print("\\n" + "=" * 50)
    print("Additional Features:")
    
    # Per-keypoint results
    print("\\n1. Per-keypoint results:")
    metric_per_kp = PercentageOfCorrectKeyPoints(
        relative_distance_threshold=0.1, 
        return_per_keypoint=True
    )
    per_kp_results = metric_per_kp.apply(y_true, y_pred)
    
    print(f"   PCK per keypoint: {per_kp_results}")
    print(f"   Average: {np.mean(per_kp_results):.4f}")
    print(f"   Best keypoint: {np.max(per_kp_results):.4f}")
    print(f"   Worst keypoint: {np.min(per_kp_results):.4f}")
    
    # Different thresholds
    print("\\n2. Different distance thresholds:")
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
    for threshold in thresholds:
        metric_t = PercentageOfCorrectKeyPoints(relative_distance_threshold=threshold)
        pck_t = metric_t.apply(y_true, y_pred)
        print(f"   PCK@{threshold:0.2f}: {pck_t:.4f}")
    
    # Detailed analysis
    print("\\n3. Detailed analysis:")
    detailed = metric.compute_detailed_results(y_true, y_pred)
    
    print(f"   Overall PCK: {detailed['overall_pck']:.4f}")
    print(f"   Distance threshold: {detailed['distance_threshold_pixels']:.1f} pixels")
    print(f"   Correct keypoints: {detailed['correct_keypoints']}/{detailed['total_keypoints']}")
    print(f"   Success rate: {detailed['correct_keypoints']/detailed['total_keypoints']*100:.1f}%")
    
    # Show some statistics
    per_sample_pck = detailed['per_sample_pck']
    print(f"   Per-sample PCK - Mean: {np.mean(per_sample_pck):.4f}")
    print(f"   Per-sample PCK - Std:  {np.std(per_sample_pck):.4f}")
    
    print("\\n" + "=" * 50)
    print("Example completed successfully!")
    print("\\nTo use in your own code:")
    print("""
    import numpy as np
    from evaluation.metrics.percentage_of_correct_keypoints import PercentageOfCorrectKeyPoints
    
    # Your data
    y_true = ...  # shape: (batch_size, h, w, num_keypoints)
    y_pred = ...  # shape: (batch_size, h, w, num_keypoints)
    
    # Create metric
    metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)
    
    # Compute PCK
    result = metric.apply(y_true, y_pred)
    print(f"PCK: {result:.4f}")
    """)

if __name__ == "__main__":
    main()