# Smart Design of Experiments (Smart DOE) - Bayesian Optimization System

## Overview

This project implements an intelligent Design-of-Experiments system based on Bayesian Optimization with Gaussian Processes. The system provides a web-based user interface for interactive optimization of multi-dimensional functions with automatic kernel selection and advanced sampling methods.

## Main Features

- **Advanced Sampling Methods**: LHS, Space-Filling LHS, Fractional Factorial, Full Factorial, Taguchi
- **Automatic Kernel Selection**: Hyperband algorithm for optimal kernel and hyperparameter selection
- **Gaussian Process Models**: Single-task and multi-task GP models with automatic scaling
- **Bayesian Optimization**: Log-Expected-Improvement acquisition function with robust optimization
- **Interactive Web Interface**: Flask-based real-time optimization interface
- **Multi-Output Support**: Weighted optimization for multiple objectives
- **Automatic Scaling**: Input normalization and output standardization

## How the System Works

### 1. **Initialization Phase**
1. **Data Generation**: Creates initial dataset using selected sampling method (LHS, Factorial, etc.)
2. **Kernel Selection**: Uses Hyperband algorithm to automatically select the best kernel type and hyperparameters
3. **Model Training**: Trains GP model with selected kernel and applies scaling transformations
4. **Optimization Setup**: Initializes Bayesian optimization loop with acquisition function

### 2. **Interactive Optimization Phase**
1. **Suggestion Generation**: Uses acquisition function to suggest next evaluation point
2. **User Input**: User provides observation value(s) for suggested parameters
3. **Model Update**: Adds new data point and retrains GP model
4. **Convergence**: System converges to optimal solution through iterative improvement

### 3. **Kernel Selection Process (Hyperband)**
The system uses the Hyperband algorithm to automatically select the best kernel:

- **Kernel Types**: RBF, Matern, Linear, Periodic, Polynomial
- **Task Kernels**: IndexKernel, ICM, LMC (for multi-task models)
- **Evaluation**: Combines Marginal Log-Likelihood and Cross-Validation
- **Resource Allocation**: Adaptive resource allocation based on dataset size
- **Early Stopping**: Efficient evaluation with early termination of poor configurations

**⚠️ Note**: Cross-validation is implemented there are still some problems that need to be solved.

## Project Structure

### Core Components

#### **Main Application**
- `run.py`: Flask application entry point
- `web_main.py`: Core business logic and model management
- `flask_app/`: Web interface and API endpoints

#### **Data Management**
- `data/create_dataset.py`: Dataset creation and management
- `data/function_factory.py`: Test functions for validation

#### **Sampling Methods**
- `sampler/lhs.py`: Latin Hypercube Sampling
- `sampler/space_filling_lhs.py`: Space-filling LHS
- `sampler/full_factorial.py`: Complete factorial designs
- `sampler/fractional_factorial.py`: Fractional factorial designs
- `sampler/taguchi.py`: Taguchi orthogonal arrays

#### **Model Components**
- `models/gp_model.py`: Main GP model class with training and prediction
- `models/gp_model_factory.py`: Factory for creating GP models
- `models/kernel_factory.py`: Kernel creation and management
- `models/likelihood_factory.py`: Likelihood function factory
- `models/mll_factory.py`: Marginal Log-Likelihood factory
- `models/optimizer_factory.py`: Optimizer factory (Adam, SGD)

#### **Kernel Selection**
- `kernel_choice/hyperband_kernel_selector.py`: Hyperband-based kernel selection

#### **Optimization**
- `optimization/bayesian_optimization_loop.py`: Main optimization loop
- `optimization/acquisition_function_factory.py`: Acquisition function factory

#### **Training**
- `training/training.py`: Traditional PyTorch training utilities

## Technical Architecture

### **Data Flow**

#### **Initialization Phase (once)**
```
User Input → Sampling → Kernel Selection → Model Training → Optimization Setup
```

#### **Iterative Optimization Phase (repeated)**
```
Optimization → Suggestions → User Observation → Model Update → Optimization
```

**Note**: Kernel selection happens only during initialization. During optimization iterations, the same kernel is used and only the model parameters are updated with new data.

### **Model Architecture**
- **Single-Task GP**: Standard Gaussian Process for single output
- **Multi-Task GP**: Multi-task GP with task kernels (IndexKernel, ICM, LMC)
- **Scaling**: Automatic input normalization and output standardization
- **Training**: Adam optimizer with convergence training

### **Optimization Strategy**
- **Acquisition Function**: Log-Expected-Improvement for robust optimization
- **Optimization**: 40 random restarts, 400 raw samples for global optimization
- **Multi-Output**: Weighted sum of objectives for multi-task problems

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

### Installation Steps

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies** (in order):
   ```bash
   pip install torch>=2.0.0
   pip install gpytorch>=1.14.0
   pip install botorch>=0.14.0
   pip install Flask>=3.0.0
   pip install numpy>=1.26.0 matplotlib>=3.8.0
   pip install pyDOE3>=1.2.0 scikit-learn>=1.3.0
   ```

3. **Start the application**:
   ```bash
   python run.py
   ```

4. **Access the interface**: Open `http://localhost:5000` in your browser

## Usage Guide

### 1. **Configuration**
- Set number of input parameters
- Define bounds for each parameter
- Choose sampling method for initial data
- Configure multi-output settings if needed

### 2. **Initialization**
- Click "Start" to begin optimization
- System automatically selects best kernel
- Initial suggestions are generated

### 3. **Interactive Optimization**
- Enter observation values for suggested parameters
- System updates model and generates new suggestions
- Continue until convergence or desired accuracy

### 4. **Results**
- Best parameters and objective values are tracked
- Optimization history is maintained

## Advanced Features

### **Multi-Output Optimization**
- Support for multiple objectives
- Weighted optimization with user-defined weights
- Task-specific kernels for correlation modeling

### **Adaptive Configuration**
- Resource allocation based on dataset size
- Automatic cross-validation configuration
- Early stopping for efficiency

### **Robust Optimization**
- Multiple random restarts for global optimization
- Acquisition function optimization with large sample sizes
- Convergence training for stable model updates

## Dependencies

### **Core Libraries**
- **PyTorch**: Deep learning framework
- **GPyTorch**: Gaussian Process implementation
- **BoTorch**: Bayesian Optimization
- **Flask**: Web framework

### **Scientific Computing**
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization
- **scikit-learn**: Cross-validation and utilities

### **Design of Experiments**
- **pyDOE3**: Design of Experiments library

## Known Limitations and Future Improvements

### **Current Limitations**
- Cross-validation in Hyperband need refinement
- Limited visualization capabilities
- Single acquisition function (Log-Expected-Improvement) -> Selection of different aquisition functions could be possible



