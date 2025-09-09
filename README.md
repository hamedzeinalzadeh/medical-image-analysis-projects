# Evaluation Package

A Python package for evaluating keypoint detection models. This package provides metrics and utilities for assessing the performance of keypoint detection algorithms, with a focus on maintainability, extensibility, and clean code practices.

## Features

- **Modular Architecture**: Built with extensible design patterns for easy addition of new metrics
- **Comprehensive Testing**: Full unit test coverage with pytest
- **Type Hints**: Complete type annotations for better code clarity and IDE support
- **Clean Code**: Well-documented code with clear variable names and docstrings
- **Flexible Configuration**: Configurable parameters for different evaluation scenarios

## Installation

### From Source

```bash
# Clone the repository (if applicable)
git clone <repository-url>
cd evaluation-package

# Install in development mode
pip install -e .
```

### Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- pytest >= 6.0.0 (for testing)

## Quick Start

```python
import numpy as np
from evaluation.metrics import PercentageOfCorrectKeyPoints

# Create sample data
batch_size, height, width, num_keypoints = 2, 100, 100, 17
y_true = np.random.rand(batch_size, height, width, num_keypoints)
y_pred = np.random.rand(batch_size, height, width, num_keypoints)

# Initialize metric
metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)

# Compute PCK
result = metric.apply(y_true, y_pred)
print(f"PCK Score: {result:.4f}")
```

## Available Metrics

### PercentageOfCorrectKeyPoints (PCK)

The PCK metric calculates the percentage of predicted keypoints that fall within a specified distance threshold from the ground truth keypoints.

#### Parameters

- `relative_distance_threshold` (float, default=0.1): Distance threshold as a fraction of image height
- `return_per_keypoint` (bool, default=False): Whether to return results per keypoint or averaged
- `name` (str, optional): Custom name for the metric

#### Usage Examples

##### Basic Usage

```python
from evaluation.metrics import PercentageOfCorrectKeyPoints
import numpy as np

# Create test data
y_true = np.zeros((1, 100, 100, 1))
y_pred = np.zeros((1, 100, 100, 1))

# Place keypoints
y_true[0, 50, 50, 0] = 1.0  # True keypoint at (50, 50)
y_pred[0, 55, 55, 0] = 1.0  # Predicted keypoint at (55, 55)

# Evaluate
metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)
pck_score = metric.apply(y_true, y_pred)
print(f"PCK: {pck_score}")  # Should be 1.0 (within threshold)
```

##### Per-Keypoint Results

```python
# Get results for each keypoint separately
metric = PercentageOfCorrectKeyPoints(
    relative_distance_threshold=0.1,
    return_per_keypoint=True
)

pck_per_keypoint = metric.apply(y_true, y_pred)
print(f"PCK per keypoint: {pck_per_keypoint}")
```

##### Detailed Analysis

```python
# Get comprehensive results
detailed_results = metric.compute_detailed_results(y_true, y_pred)

print(f"Overall PCK: {detailed_results['overall_pck']}")
print(f"Per-keypoint PCK: {detailed_results['per_keypoint_pck']}")
print(f"Distance threshold (pixels): {detailed_results['distance_threshold_pixels']}")
print(f"Correct keypoints: {detailed_results['correct_keypoints']}/{detailed_results['total_keypoints']}")
```

##### Different Thresholds

```python
# Compare different thresholds
thresholds = [0.05, 0.1, 0.15, 0.2]
for threshold in thresholds:
    metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=threshold)
    pck = metric.apply(y_true, y_pred)
    print(f"PCK@{threshold}: {pck:.4f}")
```

## Data Format

### Input Requirements

Both `y_true` and `y_pred` must be 4D numpy arrays with shape:
```
(batch_size, height, width, num_keypoints)
```

- **batch_size**: Number of samples in the batch
- **height**: Image height in pixels
- **width**: Image width in pixels  
- **num_keypoints**: Number of keypoints per sample
- **Values**: Should be in range [0, 1] representing heatmap intensities

### Keypoint Representation

Keypoints are represented as heatmaps where:
- Each keypoint has its own channel (last dimension)
- The location of maximum value in each heatmap indicates the keypoint position
- Higher values indicate higher confidence/probability

## Architecture

### Design Patterns

The package uses several design patterns for maintainability:

1. **Template Method Pattern**: `BaseMetric` defines the algorithm structure while allowing subclasses to implement specific computation logic
2. **Strategy Pattern**: Different metrics can be used interchangeably through the common interface
3. **Factory Pattern**: Easy instantiation of metrics with different configurations

### Class Hierarchy

```
BaseMetric (Abstract)
├── PercentageOfCorrectKeyPoints
└── [Future metrics can be added here]
```

### Adding New Metrics

To add a new metric, inherit from `BaseMetric`:

```python
from evaluation.metrics.base_metric import BaseMetric
import numpy as np

class MyCustomMetric(BaseMetric):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def _validate_config(self):
        if self.custom_param <= 0:
            raise ValueError("custom_param must be positive")
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Implement your metric computation here
        return 0.0  # Replace with actual computation
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=evaluation

# Run specific test file
pytest evaluation/tests/test_percentage_of_correct_keypoints.py

# Run with verbose output
pytest -v
```

### Test Structure

- `test_base_metric.py`: Tests for the abstract base class
- `test_percentage_of_correct_keypoints.py`: Comprehensive tests for PCK metric

## Development

### Code Style

The package follows these conventions:
- PEP 8 style guide
- Type hints for all public methods
- Comprehensive docstrings (Google style)
- Meaningful variable and function names
- Proper error handling with descriptive messages

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## API Reference

### BaseMetric

Abstract base class for all metrics.

#### Methods

- `apply(y_true, y_pred)`: Main method to compute the metric
- `get_config()`: Get configuration parameters
- `_validate_inputs()`: Validate input arrays (called automatically)
- `_compute_metric()`: Abstract method to implement in subclasses

### PercentageOfCorrectKeyPoints

#### Methods

- `apply(y_true, y_pred)`: Compute PCK score
- `compute_detailed_results(y_true, y_pred)`: Get comprehensive analysis
- `get_config()`: Get metric configuration

## Examples

### Complete Example

```python
import numpy as np
from evaluation.metrics import PercentageOfCorrectKeyPoints

# Simulate keypoint detection results
def create_sample_data():
    batch_size, height, width, num_keypoints = 4, 256, 256, 17
    
    # Create ground truth heatmaps
    y_true = np.zeros((batch_size, height, width, num_keypoints))
    y_pred = np.zeros((batch_size, height, width, num_keypoints))
    
    # Add some realistic keypoints
    np.random.seed(42)
    for b in range(batch_size):
        for k in range(num_keypoints):
            # Random true keypoint location
            true_y, true_x = np.random.randint(20, height-20, 2)
            y_true[b, true_y, true_x, k] = 1.0
            
            # Predicted location with some noise
            pred_y = true_y + np.random.randint(-10, 11)
            pred_x = true_x + np.random.randint(-10, 11)
            pred_y = np.clip(pred_y, 0, height-1)
            pred_x = np.clip(pred_x, 0, width-1)
            y_pred[b, pred_y, pred_x, k] = 1.0
    
    return y_true, y_pred

# Create data
y_true, y_pred = create_sample_data()

# Evaluate with different thresholds
thresholds = [0.05, 0.1, 0.15, 0.2]
print("PCK Results:")
print("-" * 40)

for threshold in thresholds:
    metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=threshold)
    pck = metric.apply(y_true, y_pred)
    print(f"PCK@{threshold:0.2f}: {pck:0.4f}")

# Detailed analysis
print("\\nDetailed Analysis:")
print("-" * 40)
metric = PercentageOfCorrectKeyPoints(relative_distance_threshold=0.1)
results = metric.compute_detailed_results(y_true, y_pred)

print(f"Overall PCK: {results['overall_pck']:0.4f}")
print(f"Distance threshold: {results['distance_threshold_pixels']:0.1f} pixels")
print(f"Correct keypoints: {results['correct_keypoints']}/{results['total_keypoints']}")
print(f"Per-keypoint PCK: {np.mean(results['per_keypoint_pck']):0.4f} ± {np.std(results['per_keypoint_pck']):0.4f}")
```

## License

[Add your license information here]

## Support

For issues, questions, or contributions, please [add contact information or repository links].