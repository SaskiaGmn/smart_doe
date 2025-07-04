# Smart Design of Experiments (Smart DOE) - Bayesian Optimization System

## Overview

This project implements an intelligent Design-of-Experiments system based on Bayesian Optimization with Gaussian Processes. The system provides a web-based user interface for interactive optimization of multi-dimensional functions.

## Main Features

- Various Sampling Methods: LHS, Space-Filling LHS, Fractional Factorial, Full Factorial, Taguchi
- Automatic Kernel Selection: Uses Hyperband algorithm for optimal kernel selection
- Gaussian Process Models: Advanced probabilistic modeling with automatic hyperparameter optimization
- Bayesian Optimization: Log-Expected-Improvement Acquisition Function
- Interactive Web Interface: Flask-based user interface for real-time optimization
- Automatic Scaling: Input normalization and output standardization


## Project Structure and File Functions

### Main Files

#### `run.py`
- Function: Main entry point of the application
- Process: 
  1. Creates Flask app with correct template and static folder configuration
  2. Registers blueprint from `flask_app/routes.py`
  3. Starts Flask server in debug mode

#### `web_main.py`
- Function: Central logic for the web application
- Main Functions:
  - `setup_first_model()`: Initializes the first GP model
  - `setup_optimization_loop()`: Configures the optimization loop
  - `get_next_optimization_iteration()`: Performs an optimization step

### Flask Application (`flask_app/`)

#### `flask_app/routes.py`
- Function: Defines all web routes and API endpoints
- Routes:
  - `/`: Main page (index.html)
  - `/initialize` (POST): Initializes the optimization model
  - `/submit_observation` (POST): Submits new observations
- Global variables: Stores GP model, optimizer and current suggestions

#### `flask_app/templates/index.html`
- Function: Main user interface
- Features:
  - Configure number of parameters and bounds
  - Select sampling method
  - Display optimization suggestions
  - Enter observations
- JavaScript handles all client-side interactions

#### `flask_app/static/`
- Function: Static files (CSS, images) - mainly functions of previous version
- Content: Styling and visualizations

### Data Management (`data/`)

#### `data/create_dataset.py`
- Class: `DatasetManager`
- Function: Creates and manages training datasets
- Main methods:
  - `func_create_dataset()`: Creates dataset based on function
  - `add_noise()`: Adds Gaussian noise
  - `setbounds()`: Sets bounds for parameters
- Sampling methods: Random, Grid, LHS, Space-Filling LHS, Fractional/Full Factorial, Taguchi

#### `data/function_factory.py`
- Class: `FunctionFactory`
- Function: Defines various test functions
- Available functions:
  - `function_xsinx()`: x * sin(x) for each dimension
  - `sum_of_sines()`: Sum of sine values
  - `multi_inputs()`: Weighted sum of squared inputs

### Sampling Methods (`sampler/`)

#### `sampler/fractional_factorial.py`
- Function: Fractional Factorial Designs
- For experiments with many factors

#### `sampler/full_factorial.py`
- Function: Complete factorial experimental design
- For Small experiments with few factors

#### `sampler/lhs.py`
- Function: Latin Hypercube Sampling
- Implementation: Uses pyDOE3 library
-  Centered criteria for better coverage

#### `sampler/space_filling_lhs.py`
- Function: Space-Filling Latin Hypercube Sampling
- Maximizes minimum distance between lhs points

#### `sampler/taguchi.py`
- Function: Taguchi Orthogonal Arrays
- Robust parameter optimization

### Models (`models/`)

#### `models/gp_model.py`
- Class: `BaseGPModel`
- Function: Main class for Gaussian Process models
- Main methods:
  - `train()`: Trains the GP model
  - `add_point_to_dataset()`: Adds new data points
  - `visualize_trained_model()`: Creates visualizations
- Features:
  - Automatic scaling (Normalize/Standardize)
  - Convergence training
  - Bounds management

#### `models/gp_model_factory.py`
- Function: Factory for GP model creation
- Supported types: SingleTaskGP, MultiTaskGP - MultiTaskGp currently not used

#### `models/kernel_factory.py`
- Class: `KernelFactory`
- Function: Creates various kernel types
- Available kernels:
  - RBF (Radial Basis Function)
  - Matern 
  - Periodic
  - Linear
  - Polynomial

#### `models/likelihood_factory.py`
- Function: Factory for likelihood functions
- Default: Gaussian Likelihood

#### `models/mll_factory.py`
- Function: Factory for Marginal Log-Likelihood
- Default: ExactMarginalLogLikelihood

#### `models/optimizer_factory.py`
- Function: Factory for optimizers
- Supported: Adam, SGD, L-BFGS-B

### Kernel Selection (`kernel_choice/`)

#### `kernel_choice/hyperband_kernel_selector.py`
- Class: `HyperbandKernelSelector`
- Function: Automatic kernel selection with Hyperband algorithm (Based on https://arxiv.org/pdf/1603.06560)
- Features:
  - Adaptive resource allocation
  - Cross-validation and MLL evaluation
  - LHS-based hyperparameter search
  - Early stopping
- Evaluation metrics:
  - Marginal Log-Likelihood (70% weighting)
  - K-Fold Cross-Validation (30% weighting)

### Optimization (`optimization/`)

#### `optimization/bayesian_optimization_loop.py`
- Class: `BayesianOptimizationLoop`
- Function: Main class for Bayesian Optimization
- Main methods:
  - `update_model()`: Updates model with new data
  - `optimization_iteration()`: Performs optimization step
  - `get_optimization_status()`: Returns optimization status
- Features:
  - Log-Expected-Improvement Acquisition Function
  - Automatic best value tracking
  - Multiple restarts for robust optimization

#### `optimization/acquisition_function_factory.py`
- Function: Factory for acquisition functions
- Available types: LogExpectedImprovement, ExpectedImprovement, UCB

### Training (`training/`)

#### `training/training.py`
- Function: Traditional PyTorch training
- Alternative to botorch fit_gpytorch_model
- Adam optimizer with configurable epochs

### Visualization (`visualization/`)

#### `visualization/visualization.py`
- Class: `GPVisualizer`
- Function: Creates visualizations for GP models
- Features:
  - Partial Dependence Plots
  - Uncertainty visualization
  - Model performance diagrams

### Utilities (`utils/`)

#### `utils/conversion_utils.py`
- Function: Converts Matplotlib plots to PNG
- Usage: For web integration

#### `utils/checking_utils.py`
- Function: Validation functions for data and parameters

#### `utils/scaling_utils.py`
- Function: Scaling functions for data preprocessing

## Detailed Process Flow

### 1. Application Start (`run.py`)
```
1. Flask app is created
2. Blueprint is registered
3. Server starts on localhost:5000
```

### 2. User Initialization (Web Interface)
```
1. User opens browser and goes to localhost:5000
2. Configures number of parameters 
3. Selects sampling method:
   - LHS (default)
   - Space-Filling LHS
   - Fractional Factorial (with main factors to be chosen)
   - Full Factorial
   - Taguchi
4. Defines bounds for each parameter
5. Clicks "Start" → POST request to /initialize
```

### 3. Model Initialization (`/initialize` Route)
```
1. Receives configuration data from frontend
2. Calls setup_first_model():
   1. Creates DatasetManager with chosen dimensions
   2. Generates initial data points with chosen sampling method
   3. Applies chosen test function (multi_inputs)
   4. Adds noise (0.1 standard deviation)
3. Automatic kernel selection:
   1. HyperbandKernelSelector is initialized
   2. Tests various kernel types and hyperparameters
   3. Evaluates with MLL (70%) and CV (30%)
   4. Selects best kernel
4. GP model is created and trained:
   1. BaseGPModel with chosen kernel
   2. Gaussian Likelihood
   3. Input normalization, output standardization
   4. Training over 100 epochs with L-BFGS-B
5. Optimization loop is set up:
   1. BayesianOptimizationLoop is created
   2. First suggestions are generated
   3. Status is returned to frontend
```

### 4. Interactive Optimization (`/submit_observation` Route)
```
1. User enters observation value
2. POST request to /submit_observation
3. update_model() is called:
   1. New data point is added to dataset
   2. Best value is updated if necessary
   3. GP model is retrained
4. Next optimization step:
   1. Acquisition function is optimized
   2. New suggestions are generated
   3. Status is updated
5. Results are returned to frontend
```

### 5. Optimization Algorithm (Bayesian Optimization)
```
1. Acquisition Function (Log-Expected-Improvement):
   - Calculates expected improvement
   - Considers model uncertainty
   - Selects most promising parameter combination

2. Acquisition Function Optimization:
   - 40 random restarts
   - 400 raw samples
   - L-BFGS-B optimizer

3. Model Update:
   - New data point is added
   - GP model is retrained
   - Hyperparameters are adjusted
```

## Technical Details

### Dependencies
- PyTorch: Deep learning framework
- GPyTorch: Gaussian Process implementation
- BoTorch: Bayesian Optimization
- Flask: Web framework
- pyDOE3: Design of Experiments
- scikit-learn: Cross-validation
- matplotlib: Visualization

### Data Formats
- Input: `torch.Tensor` with shape `[n_samples, n_dimensions]`
- Output: `torch.Tensor` with shape `[n_samples, 1]`
- Bounds: `torch.Tensor` with shape `[2, n_dimensions]` (min/max per dimension)

### Scaling
- Input: Normalization (0-1 scale)
- Output: Standardization (mean=0, std=1)
- Automatic adjustment: When new data points are added

### Kernel Types
1. RBF: Radial Basis Function (default)
2. Matern: For less smooth functions
3. Periodic: For periodic functions
4. Linear: For linear trends
5. Polynomial: For polynomial relationships

### Sampling Methods
1. LHS: Latin Hypercube Sampling (default)
2. Space-Filling LHS: Maximizes minimum distance
3. Fractional Factorial: For many factors
4. Full Factorial: Complete combinations
5. Taguchi: Orthogonal arrays

## Usage

### Installation

#### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

#### Setup Steps

1. Create and activate virtual environment:

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
2. Install dependencies (recommended order):

   pip install torch==2.0.1
   pip install gpytorch==1.11.0
   pip install botorch==0.10.0
   


### To Start:

python run.py

Open `http://localhost:5000` in your browser

1. Configuration: Set number of parameters and bounds
2. Sampling: Generate initial data points
3. Optimization: Interactively enter observations
4. Convergence: System converges to optimum


