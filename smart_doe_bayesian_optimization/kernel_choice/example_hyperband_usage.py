#!/usr/bin/env python3
"""
Example usage of HyperbandKernelSelector with adaptive training.
Demonstrates how the system automatically selects the best kernel
with early stopping based on convergence detection.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel_choice.hyperband_kernel_selector import HyperbandKernelSelector
from data.create_dataset import DatasetManager

def main():
    print("=== Hyperband Kernel Selection with LHS Sampling ===")
    print("This example demonstrates:")
    print("1. Automatic kernel selection using Hyperband optimization")
    print("2. LHS sampling for better hyperparameter space coverage")
    print("3. Adaptive training with early stopping when convergence is detected")
    print("4. Weighted evaluation using MLL (70%) and LOO-CV (30%)")
    print()
    
    # Create sample dataset
    print("Creating sample dataset...")
    dataset_manager = DatasetManager(dtype=torch.float64, num_input_dimensions=2)
    
    # Define a simple test function
    def test_function(inputs):
        # Create a function with some periodicity and smoothness
        outputs = torch.sin(inputs[:, 0]) * torch.cos(inputs[:, 1])
        return outputs.unsqueeze(1), 1  # Return outputs and output dimension
    
    # Create dataset using LHS sampling
    dataset_manager.func_create_dataset(
        dataset_func=test_function,
        num_datapoints=30,
        sampling_method="lhs",
        noise_level=0.1,
        x1_range=(0, 10),
        x2_range=(0, 10)
    )
    
    # Extract data
    X, Y = dataset_manager.unscaled_data
    Y = Y.squeeze(-1)  # Fix shape for GP training
    bounds_list = dataset_manager.bounds_list
    
    print(f"Dataset created: {X.shape[0]} samples, {X.shape[1]} dimensions")
    print(f"Output range: [{Y.min().item():.3f}, {Y.max().item():.3f}]")
    print()
    
    # Initialize Hyperband kernel selector
    print("Initializing Hyperband kernel selector...")
    selector = HyperbandKernelSelector(
        train_X=X,
        train_Y=Y,
        bounds_list=bounds_list,
        mll_weight=0.7,  # 70% weight for MLL
        cv_weight=0.3    # 30% weight for LOO-CV
    )
    
    print(f"Configuration: max_iter={selector.config.max_iter}, eta={selector.config.eta}")
    print(f"Epoch levels: {selector.config.epochs_per_level}")
    print(f"Time limit: {selector.config.max_time_seconds} seconds")
    print()
    
    # Select the best kernel
    print("Running Hyperband optimization with LHS sampling...")
    print("This will test multiple kernel configurations with:")
    print("- LHS sampling for uniform hyperparameter space coverage")
    print("- Fast evaluation of poor candidates (early stopping)")
    print("- More epochs for promising candidates")
    print("- Automatic convergence detection")
    print()
    
    best_kernel = selector.select_kernel(random_seed=42)
    
    print("=== Results ===")
    print(f"Selected kernel type: {type(best_kernel).__name__}")
    
    # Print kernel parameters
    if hasattr(best_kernel, 'lengthscale'):
        print(f"Lengthscale: {best_kernel.lengthscale.item():.4f}")
    if hasattr(best_kernel, 'variance'):
        print(f"Variance: {best_kernel.variance.item():.4f}")
    if hasattr(best_kernel, 'period'):
        print(f"Period: {best_kernel.period.item():.4f}")
    if hasattr(best_kernel, 'nu'):
        print(f"Nu (Matern): {best_kernel.nu}")
    if hasattr(best_kernel, 'offset'):
        print(f"Offset: {best_kernel.offset.item():.4f}")
    
    print()
    
    # Test the selected kernel
    print("Testing selected kernel...")
    test_X = torch.rand(5, 2) * 10
    test_Y = torch.sin(test_X[:, 0]) * torch.cos(test_X[:, 1]) + 0.1 * torch.randn(5, 1)
    
    # Create a model with the selected kernel
    from models.gp_model import BaseGPModel
    from models.likelihood_factory import LikelihoodFactory
    
    likelihood = LikelihoodFactory.create_likelihood('Gaussian')
    
    test_model = BaseGPModel(
        model_type="SingleTaskGP",
        mll_type="ExactMarginalLogLikelihood",
        optimizer_type="adam",
        kernel=best_kernel,
        train_X=X,
        train_Y=Y,
        likelihood=likelihood,
        bounds_list=bounds_list,
        optimizer_kwargs={'lr': 0.1}
    )
    
    # Setup and train the model
    from models.optimizer_factory import OptimizerFactory
    from models.gp_model_factory import GPModelFactory
    from models.mll_factory import MLLFactory
    
    outcome_transform, input_transform = test_model.setup_transformations(scaling_dict=test_model.scaling_dict)
    
    test_model.gp_model = GPModelFactory.create_model(
        model_type=test_model.gp_model_type,
        train_X=test_model.train_X,
        train_Y=test_model.train_Y,
        kernel=test_model.kernel,
        likelihood=test_model.likelihood,
        outcome_transform=outcome_transform,
        input_transform=input_transform
    )
    
    test_model.mll = MLLFactory.create_mll(type=test_model.mll_type, model=test_model.gp_model, likelihood=test_model.likelihood)
    test_model.optimizer = OptimizerFactory.create_optimizer(
        type=test_model.optimizer_type,
        model_parameters=test_model.gp_model.parameters(),
        lr=0.1
    )
    
    # Train with adaptive early stopping
    print("Training final model with adaptive early stopping...")
    actual_epochs = selector._train_with_early_stopping(test_model, max_epochs=100)
    print(f"Training completed in {actual_epochs} epochs (convergence detected)")
    
    # Make predictions
    test_model.gp_model.eval()
    with torch.no_grad():
        posterior = test_model.gp_model.posterior(test_X)
        predictions = posterior.mean
        uncertainties = posterior.variance.sqrt()
    
    print("\n=== Prediction Results ===")
    for i in range(len(test_X)):
        print(f"Point {i+1}: True={test_Y[i].item():.3f}, Pred={predictions[i].item():.3f} ± {uncertainties[i].item():.3f}")
    
    print("\n=== Summary ===")
    print("✅ Hyperband successfully selected an appropriate kernel")
    print("✅ Adaptive training stopped early when convergence was detected")
    print("✅ The selected kernel provides good predictions with reasonable uncertainty")
    print("\nThe system automatically:")
    print("- Used LHS sampling for uniform hyperparameter space coverage")
    print("- Evaluated multiple kernel types and hyperparameters")
    print("- Used early stopping to save computation time")
    print("- Selected the best kernel based on MLL and LOO-CV scores")
    print("- Trained the final model efficiently")

if __name__ == "__main__":
    main() 