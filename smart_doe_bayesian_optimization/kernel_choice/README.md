# Automatic Kernel Selection for Gaussian Processes

Minimal, efficient automatic kernel selection using **Marginal Log-Likelihood (MLL)** and **Leave-One-Out Cross-Validation (LOO-CV)**.

## Overview

A streamlined system that automatically selects the optimal kernel for your Gaussian Process model with minimal overhead.

**Core Approach**: `Weighted Score = 0.7 × MLL + 0.3 × LOO-CV`

## Quick Integration

### **One-Line Integration**
```python
from kernel_choice.automatic_kernel_selector import AutomaticKernelSelector

# Replace manual kernel selection with automatic selection
best_kernel = AutomaticKernelSelector(train_X, train_Y, bounds_list).select_kernel()

# Use in your existing code
gp_model = BaseGPModel(..., kernel=best_kernel, ...)
```

### **With Custom Weights**
```python
# Adjust balance between fit and generalization
best_kernel = AutomaticKernelSelector(
    train_X, train_Y, bounds_list,
    mll_weight=0.8,  # More emphasis on fit
    cv_weight=0.2    # Less emphasis on generalization
).select_kernel()
```

## What It Does

1. **Tests 8 kernel configurations** automatically
2. **Computes MLL and LOO-CV** for each kernel
3. **Selects the best kernel** based on weighted score
4. **Returns the optimal kernel** for your GP model

## Available Kernels

- **RBF**: 3 configurations (lengthscales: 0.5, 1.0, 2.0)
- **Matern**: 2 configurations (ν=1.5, ν=2.5)
- **Linear**: 1 configuration
- **Periodic**: 1 configuration
- **Polynomial**: 1 configuration

## Weight Guidelines

| Use Case | MLL Weight | CV Weight | Description |
|----------|------------|-----------|-------------|
| **Optimization** | 0.8 | 0.2 | Emphasize fit quality |
| **Balanced** | 0.7 | 0.3 | Default, good for most cases |
| **Prediction** | 0.3 | 0.7 | Emphasize generalization |

## Integration Examples

### **Before (Manual Selection)**
```python
# Manual kernel selection
kernel = KernelFactory.create_kernel('Matern', nu=2.5)
gp_model = BaseGPModel(..., kernel=kernel, ...)
```

### **After (Automatic Selection)**
```python
# Automatic kernel selection
best_kernel = AutomaticKernelSelector(train_X, train_Y, bounds_list).select_kernel()
gp_model = BaseGPModel(..., kernel=best_kernel, ...)
```

## Performance

- **Silent operation**: No verbose output
- **Fast execution**: Optimized for minimal overhead
- **Robust selection**: Uses proven statistical methods
- **Memory efficient**: Minimal memory footprint

## Dependencies

- `torch` and `gpytorch` (for GP models)
- `numpy` (for numerical operations)

## Why This Approach?

- **Simple**: Clear, understandable metrics
- **Reliable**: LOO-CV prevents overfitting
- **Flexible**: Adjustable weights for different applications
- **Efficient**: Minimal computational overhead 

# Kernel Choice Module

This module provides automatic kernel selection for Gaussian Processes using different optimization strategies.

## Features

### 1. Hyperband Kernel Selector (Recommended)
- **Efficient optimization** using Hyperband algorithm
- **Joint optimization** of kernel type and hyperparameters
- **Adaptive resource allocation** based on dataset size
- **Robust evaluation** using MLL (70%) + LOO-CV (30%)
- **Early stopping** and time limits for efficiency

### 2. Simple Kernel Evaluator
- **Basic evaluation** of fixed kernel configurations
- **MLL and LOO-CV** scoring
- **Manual kernel selection**

## Quick Start

### Hyperband Kernel Selection (Recommended)

```python
from data.create_dataset import create_dataset
from hyperband_kernel_selector import HyperbandKernelSelector

# Create your dataset
train_X, train_Y, bounds_list, scaling_dict = create_dataset(
    function_name="ackley",
    n_samples=30,
    noise_level=0.1
)

# Initialize selector
selector = HyperbandKernelSelector(
    train_X=train_X,
    train_Y=train_Y,
    bounds_list=bounds_list,
    scaling_dict=scaling_dict,
    mll_weight=0.7,  # 70% MLL
    cv_weight=0.3    # 30% LOO-CV
)

# Select best kernel
best_kernel = selector.select_kernel(random_seed=42)
```

### Simple Kernel Evaluation

```python
from kernel_evaluator import KernelEvaluator
from models.kernel_factory import KernelFactory

# Create evaluator
evaluator = KernelEvaluator(train_X, train_Y, bounds_list, scaling_dict)

# Evaluate specific kernels
kernels = [
    KernelFactory.create_kernel('RBF'),
    KernelFactory.create_kernel('Matern'),
    KernelFactory.create_kernel('Linear')
]

best_kernel = evaluator.select_best_kernel(kernels)
```

## Configuration

### Hyperband Parameters

The system automatically adapts based on dataset size:

**Small Datasets (< 50 points):**
- Max iterations: 50
- Epochs per level: [10, 25, 50]
- Time limit: 5 minutes

**Large Datasets (≥ 50 points):**
- Max iterations: 100
- Epochs per level: [10, 25, 50, 100]
- Time limit: 10 minutes

### Supported Kernels

1. **RBF Kernel**
   - Parameters: lengthscale, variance
   - Range: lengthscale [0.1, 10.0], variance [0.1, 5.0]

2. **Matern Kernel**
   - Parameters: nu, lengthscale, variance
   - Values: nu ∈ {1.5, 2.5}, lengthscale [0.1, 10.0], variance [0.1, 5.0]

3. **Linear Kernel**
   - Parameters: variance, offset
   - Range: variance [0.1, 5.0], offset [-2.0, 2.0]

4. **Periodic Kernel**
   - Parameters: lengthscale, period, variance
   - Range: lengthscale [0.1, 10.0], period [0.1, 10.0], variance [0.1, 5.0]

5. **Polynomial Kernel**
   - Parameters: power, variance, offset
   - Values: power ∈ {2, 3}, variance [0.1, 5.0], offset [-2.0, 2.0]

## Evaluation Metrics

### Marginal Log-Likelihood (MLL)
- Measures how well the kernel explains the training data
- Higher values indicate better fit
- Weight: 70% (default)

### Leave-One-Out Cross-Validation (LOO-CV)
- Measures predictive performance
- More robust to overfitting
- Weight: 30% (default)

## Error Handling

- **Non-converging kernels** are automatically assigned worst scores
- **Numerical issues** (NaN/Inf) are handled gracefully
- **Time limits** prevent infinite optimization
- **Early stopping** when no improvement is found

## Integration

The selected kernel can be directly used in your Bayesian optimization pipeline:

```python
# Use the selected kernel in your GP model
from models.gp_model import BaseGPModel

gp_model = BaseGPModel(
    model_type="SingleTaskGP",
    kernel=best_kernel,  # Your selected kernel
    train_X=train_X,
    train_Y=train_Y,
    # ... other parameters
)
```

## Example Usage

Run the example script to see Hyperband kernel selection in action:

```bash
python example_hyperband_usage.py
```

This will:
1. Create a sample dataset
2. Run Hyperband optimization
3. Select the best kernel
4. Test the selected kernel 