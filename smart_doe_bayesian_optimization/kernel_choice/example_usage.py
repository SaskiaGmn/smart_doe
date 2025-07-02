"""
Minimal example usage of the automatic kernel selection system.

This script demonstrates the simple integration of automatic kernel selection
into your existing GP workflow.
"""

import torch
from data.create_dataset import DatasetManager
from data.function_factory import FunctionFactory
from kernel_choice.automatic_kernel_selector import AutomaticKernelSelector
from models.gp_model import BaseGPModel
from models.likelihood_factory import LikelihoodFactory
from gpytorch.constraints import GreaterThan

def minimal_integration_example():
    """Minimal example showing how to integrate automatic kernel selection."""
    
    # Step 1: Create your dataset (your existing code)
    dataset = DatasetManager(dtype=torch.float64, num_input_dimensions=2)
    dataset.func_create_dataset(
        FunctionFactory.multi_inputs,
        num_datapoints=20,
        sampling_method="lhs",
        noise_level=0.1,
        x1_range=(0, 10),
        x2_range=(0, 10)
    )
    
    # Step 2: Automatically select the best kernel (new step)
    selector = AutomaticKernelSelector(
        train_X=dataset.unscaled_data[0],
        train_Y=dataset.unscaled_data[1],
        bounds_list=dataset.bounds_list,
        mll_weight=0.7,  # 70% weight for MLL
        cv_weight=0.3    # 30% weight for LOO-CV
    )
    
    best_kernel = selector.select_kernel(training_epochs=50)
    
    # Step 3: Use the selected kernel in your existing GP model (your existing code)
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
        best_kernel,  # Use the automatically selected kernel
        dataset.unscaled_data[0],
        dataset.unscaled_data[1],
        likelihood,
        bounds_list=dataset.bounds_list,
        scaling_dict=scaling_dict
    )
    
    # Train your model (your existing code)
    gp_model.train(num_epochs=100)
    
    return gp_model

def one_line_integration():
    """Even more minimal - one line integration."""
    
    # Your existing dataset
    dataset = DatasetManager(dtype=torch.float64, num_input_dimensions=2)
    dataset.func_create_dataset(
        FunctionFactory.multi_inputs,
        num_datapoints=20,
        sampling_method="lhs",
        noise_level=0.1,
        x1_range=(0, 10),
        x2_range=(0, 10)
    )
    
    # One line to get the best kernel
    best_kernel = AutomaticKernelSelector(
        dataset.unscaled_data[0], 
        dataset.unscaled_data[1], 
        dataset.bounds_list
    ).select_kernel()
    
    # Use in your existing code
    likelihood = LikelihoodFactory.create_likelihood('Gaussian')
    scaling_dict = {'input': 'normalize', 'output': 'standardize'}
    
    gp_model = BaseGPModel(
        "SingleTaskGP",
        "ExactMarginalLogLikelihood",
        "adam",
        best_kernel,  # Automatically selected kernel
        dataset.unscaled_data[0],
        dataset.unscaled_data[1],
        likelihood,
        bounds_list=dataset.bounds_list,
        scaling_dict=scaling_dict
    )
    
    gp_model.train(num_epochs=100)
    
    return gp_model

if __name__ == "__main__":
    # Run minimal examples
    try:
        print("Testing minimal integration...")
        gp_model1 = minimal_integration_example()
        print("✓ Minimal integration successful")
        
        print("Testing one-line integration...")
        gp_model2 = one_line_integration()
        print("✓ One-line integration successful")
        
        print("\nBoth examples completed successfully!")
        print("The automatic kernel selection is now ready for your workflow.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 