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

def setup_first_model(num_dimensions: int = 3, bounds: dict = None, sampling_method: str = 'lhs', main_factors: int = None, num_outputs: int = 1, output_weights: list = None, task_kernel_type: str = 'IndexKernel'):
    try:
        # Create a multi-dimensional dataset
        dataset = DatasetManager(dtype=torch.float64, num_input_dimensions=num_dimensions, num_output_dimensions=num_outputs)
        
        # Check if bounds are provided
        if bounds is None:
            raise ValueError("Bounds must be provided")
        ranges = bounds
        
        # Choose function based on number of outputs
        if num_outputs == 1:
            dataset_func = FunctionFactory.multi_inputs
        else:
            dataset_func = FunctionFactory.multi_output_quadratic
        
        dataset.func_create_dataset(
            dataset_func,
            num_datapoints=15,
            sampling_method=sampling_method,
            noise_level=0.1,
            main_factors=main_factors,
            **ranges
        )

        train_X, train_Y = dataset.unscaled_data
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)  # [n] -> [n, 1]
        
        # Use Hyperband to automatically select the best kernel
        print("Starting automatic kernel selection with Hyperband...")
        selector = HyperbandKernelSelector(
            train_X=train_X,
            train_Y=train_Y,
            bounds_list=dataset.bounds_list,
            mll_weight=0.7,
            cv_weight=0.3,
            num_outputs=num_outputs
        )
        
        # Select the best kernel
        best_kernel = selector.select_kernel(random_seed=42)
        print(f"Selected kernel: {type(best_kernel).__name__}")
        print(f"Kernel: {best_kernel}")
        
        # Get the best task kernel type for multi-task models
        task_kernel_type = None  # Single-task needs no task kernel
        if num_outputs > 1:
            task_kernel_type = selector.get_best_task_kernel_type()
            print(f"Task kernel type: {task_kernel_type}")

        # Choose likelihood based on number of outputs
        if num_outputs == 1:
            likelihood = LikelihoodFactory.create_likelihood(
                'Gaussian',
                noise_constraint=GreaterThan(1e-5)
            )
        else:
            likelihood = LikelihoodFactory.create_likelihood(
                'Multitask',
                num_tasks=num_outputs,
                noise_constraint=GreaterThan(1e-5)
            )

        scaling_dict = {
            'input': 'normalize',
            'output': 'standardize'
        }

        # Choose model type based on number of outputs
        model_type = "MultiTaskGP" if num_outputs > 1 else "SingleTaskGP"

        gp_model = BaseGPModel(
            model_type,
            "ExactMarginalLogLikelihood",
            "adam",
            best_kernel,
            train_X,
            train_Y,
            likelihood,
            bounds_list=dataset.bounds_list,
            scaling_dict=scaling_dict,
            optimizer_kwargs={"lr": 0.1},
            num_outputs=num_outputs,
            task_kernel_type=task_kernel_type
        )

        gp_model.train(num_epochs=100)
        return gp_model
    except Exception as e:
        print(f"Error in setup_first_model: {str(e)}")
        raise

def setup_optimization_loop(gp_model, acq_func_type: str = "LogExp_Improvement", is_maximization: bool = True, output_weights: list = None, optimization_directions: list = None):
    """Sets up the optimization loop.
    
    Args:
        gp_model: The trained GP model
        acq_func_type: Type of acquisition function
        is_maximization: Whether to maximize or minimize (legacy parameter, kept for compatibility)
        output_weights: List of weights for each output (for multi-output optimization)
        optimization_directions: List of optimization directions for each output ('maximize' or 'minimize')
        
    Returns:
        gp_optimizer: The optimizer
        next_value: Tensor with the suggested input parameters
    """
    try:
        # For multi-output models, use weighted sum acquisition function
        if gp_model.is_multi_task:
            if acq_func_type == "LogExp_Improvement":
                acq_func_type = "Weighted_Sum"
            
            # Convert weights to tensor if provided
            if output_weights is None:
                raise ValueError("output_weights must be provided for multi-output optimization")
            weights = torch.tensor(output_weights, dtype=torch.float32)
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            
            # Set default optimization directions if not provided
            if optimization_directions is None:
                optimization_directions = ['maximize'] * len(output_weights)
            
            gp_optimizer = BayesianOptimizationLoop(
                base_model=gp_model,
                acq_func_type=acq_func_type,
                is_maximization=is_maximization,
                acq_func_kwargs={'weights': weights},
                optimization_directions=optimization_directions
            )
        else:
            # Single-output optimization
            # Use the optimization direction for single output
            if optimization_directions is not None and len(optimization_directions) > 0:
                is_maximization = optimization_directions[0] == 'maximize'
            
            gp_optimizer = BayesianOptimizationLoop(
                base_model=gp_model,
                acq_func_type=acq_func_type,
                is_maximization=is_maximization,
                optimization_directions=optimization_directions
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
        input_value: A single value (float) - the output (for single-output) or list of values (for multi-output)
        original_x: Tensor with the input parameters
        
    Returns:
        next_value: Tensor with the next suggested input parameters
    """
    try:
        # Ensure that original_x has the correct shape (2D)
        if original_x.dim() == 1:
            original_x = original_x.unsqueeze(0)  # [n] -> [1, n]
            
        # Convert input_value to appropriate format
        if isinstance(input_value, (list, tuple)):
            # Multi-output: convert list to tensor
            input_value = torch.tensor([input_value], dtype=torch.float32)
        else:
            # Single-output: convert to tensor
            input_value = torch.tensor([[input_value]], dtype=torch.float32)
            
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

