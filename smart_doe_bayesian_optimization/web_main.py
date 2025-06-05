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
from optimization.optimization import GPOptimizer
import matplotlib.pyplot as plt
from gpytorch.priors.torch_priors import GammaPrior
from utils.conversion_utils import matplotlib_to_png
from optimization.bayesian_optimization_loop import BayesianOptimizationLoop

def setup_first_model(num_dimensions: int = 3, bounds: dict = None, sampling_method: str = 'lhs'):
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
            num_datapoints=5,
            sampling_method=sampling_method,
            noise_level=0.1,
            **ranges
        )

        # Create a Matern kernel with ARD for multiple dimensions
        kernel = KernelFactory.create_kernel(
            'Matern',
            nu=2.5,
            lengthscale_prior=GammaPrior(3.0, 6.0)
        )

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
            kernel,
            dataset.unscaled_data[0],
            dataset.unscaled_data[1],
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

def setup_optimization_loop(gp_model):
    """Setzt die Optimierungsschleife auf.
    
    Args:
        gp_model: Das trainierte GP-Modell
        
    Returns:
        gp_optimizer: Der Optimizer
        next_value: Tensor mit den vorgeschlagenen Input-Parametern
    """
    try:
        gp_optimizer = BayesianOptimizationLoop(gp_model)
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
    """Führt einen Optimierungsschritt durch.
    
    Args:
        optimizer: Der Optimizer
        input_value: Ein einzelner Wert (float) - der Output
        original_x: Tensor mit den Input-Parametern
        
    Returns:
        next_value: Tensor mit den nächsten vorgeschlagenen Input-Parametern
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