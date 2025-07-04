import models
import torch
from models.gp_model import BaseGPModel
from models.kernel_factory import KernelFactory
from gpytorch.priors import NormalPrior
from models.likelihood_factory import LikelihoodFactory
from data.create_dataset import DatasetManager
from data.function_factory import FunctionFactory
from gpytorch.constraints import GreaterThan
from models.optimizer_factory import OptimizerFactory

import matplotlib.pyplot as plt
from gpytorch.priors.torch_priors import GammaPrior
from optimization.bayesian_optimization_loop import BayesianOptimizationLoop
from kernel_choice.hyperband_kernel_selector import HyperbandKernelSelector

def setup_first_model(num_dimensions: int = 3, bounds: dict = None, sampling_method: str = 'lhs', main_factors: int = None):
    try:
        # Create a multi-dimensional dataset
        dataset = DatasetManager(dtype=torch.float64, num_input_dimensions=num_dimensions)
        
        # Use provided bounds or default ranges
        if bounds is None:
            ranges = {f'x{i+1}_range': (0, 10) for i in range(num_dimensions)}
        else:
            ranges = bounds
        
        dataset.func_create_dataset(
            FunctionFactory.multi_inputs,
            num_datapoints=10,
            sampling_method=sampling_method,
            noise_level=0.1,
            main_factors=main_factors,
            **ranges
        )

        # Extract data and ensure correct shape
        train_X, train_Y = dataset.unscaled_data
        # Ensure train_Y has shape (n, 1) for GP training
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)  # [n] -> [n, 1]
        
        # Use Hyperband to automatically select the best kernel
        print("Starting automatic kernel selection with Hyperband...")
        selector = HyperbandKernelSelector(
            train_X=train_X,
            train_Y=train_Y,
            bounds_list=dataset.bounds_list,
            mll_weight=0.7,
            cv_weight=0.3
        )
        
        # Select the best kernel
        best_kernel = selector.select_kernel(random_seed=42)
        print(f"Kernel ausgewählt: {type(best_kernel).__name__}")
        print(f"Kernel: {best_kernel}")

        likelihood = LikelihoodFactory.create_likelihood(
            'Gaussian',
            noise_constraint=GreaterThan(1e-5)
        )

        scaling_dict = {
            'input': 'normalize',
            'output': 'standardize'
        }

        gp_model = BaseGPModel(
            "SingleTaskGP",
            "ExactMarginalLogLikelihood",
            "adam",
            best_kernel,
            train_X,
            train_Y,
            likelihood,
            bounds_list=dataset.bounds_list,
            scaling_dict=scaling_dict,
            optimizer_kwargs={"lr": 0.1}
        )

        gp_model.train(num_epochs=100)
        return gp_model
    except Exception as e:
        print(f"Error in setup_first_model: {str(e)}")
        raise

def setup_optimization_loop(gp_model, acq_func_type: str = "LogExp_Improvement", is_maximization: bool = True):
    """Sets up the optimization loop.
    
    Args:
        gp_model: The trained GP model
        acq_func_type: Type of acquisition function
        is_maximization: Whether to maximize or minimize
        
    Returns:
        gp_optimizer: The optimizer
        next_value: Tensor with the suggested input parameters
    """
    try:
        gp_optimizer = BayesianOptimizationLoop(
            base_model=gp_model,
            acq_func_type=acq_func_type,
            is_maximization=is_maximization
        )
        # Get the first suggestion
        next_value, _ = gp_optimizer.optimization_iteration(num_restarts=40, raw_samples=400)
        # Ensure that next_value has the correct shape (2D)
        if next_value.dim() == 1:
            next_value = next_value.unsqueeze(0)  # [n] -> [1, n]
        return gp_optimizer, next_value
    except Exception as e:
        print(f"Error in setup_optimization_loop: {str(e)}")
        raise

def get_next_optimization_iteration(optimizer, input_value, original_x):
    """Performs an optimization step.
    
    Args:
        optimizer: The optimizer
        input_value: A single value (float) - the output
        original_x: Tensor with the input parameters
        
    Returns:
        next_value: Tensor with the next suggested input parameters
    """
    try:
        # Ensure that original_x has the correct shape (2D)
        if original_x.dim() == 1:
            original_x = original_x.unsqueeze(0)  # [n] -> [1, n]
            
        # Update the model with the new observation
        optimizer.update_model(input_value, original_x)
        
        # Get the next suggestion
        next_value, _ = optimizer.optimization_iteration(num_restarts=40, raw_samples=400)
        
        # Ensure that next_value has the correct shape (2D)
        if next_value.dim() == 1:
            next_value = next_value.unsqueeze(0)  # [n] -> [1, n]
            
        return next_value
    except Exception as e:
        print(f"Error in get_next_optimization_iteration: {str(e)}")
        raise

