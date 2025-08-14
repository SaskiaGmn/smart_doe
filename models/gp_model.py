from models.gp_model_factory import GPModelFactory
from models.kernel_factory import KernelFactory
from gpytorch.kernels import Kernel
from models.mll_factory import MLLFactory
from models.optimizer_factory import OptimizerFactory
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
import gpytorch
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.transforms.input import InputTransform, Normalize
import torch
from typing import Dict, Optional, List, Tuple

#
# TODO: for the optimization iterations: it should be considered, that the train_X and train_Y data only should be updated, when the model is also updated for it. 
#       -> avoid the case, that the state of the data (X and Y) is different from the model state -> will result in problems for the visualization script


'''
Shape of the bounds list: 
torch.Size([2, d]) - ([min_x1, min_x2, ...],[max_x1, max_x2, ...])

tensor([[  0,  10, 100],[  6,  20, 200]]) for [(0, 6), (10, 20), (100, 200)]

Available rescaling methods:
    - Input: "normalize" (Normalize transform)
    - Output: "standardize" (Standardize transform)
'''

class BaseGPModel():
    """
    Base class for Gaussian Process models supporting both single-task and multi-task scenarios.
    
    This class handles model creation, training, and data management for GP models
    with configurable kernels, likelihoods, and transformations.
    """
    
    def __init__(self, model_type: str, mll_type: str, optimizer_type: str, kernel: Kernel, train_X: torch.Tensor, train_Y: torch.Tensor, likelihood: Likelihood, bounds_list: List[Tuple], scaling_dict: dict = None, optimizer_kwargs: dict=None, num_outputs: int = 1, task_kernel_type: str = 'IndexKernel'):
        """
        Initialize the GP model with training data and configuration.
        
        Args:
            model_type: Type of GP model ('SingleTaskGP', 'MultiTaskGP', etc.)
            mll_type: Marginal log likelihood type
            optimizer_type: Optimizer type for training
            kernel: GP kernel function
            train_X: Training input data
            train_Y: Training output data
            likelihood: GP likelihood function
            bounds_list: List of (lower, upper) bounds for each input dimension
            scaling_dict: Dictionary specifying input/output scaling methods
            optimizer_kwargs: Additional optimizer parameters
            num_outputs: Number of output dimensions (1 for single-task)
            task_kernel_type: Type of task kernel for multi-task models
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.kernel = kernel
        self.likelihood = likelihood
        self.bounds_list = self.create_bounds_tensor(bounds_list=bounds_list)
        self.gp_model_type = model_type
        self.scaling_dict = scaling_dict
        self.outcome_transform, self.input_transform = None, None
        self.gp_model = None
        self.mll_type = mll_type
        self.mll = None
        self.optimizer_type = optimizer_type
        self.optimizer = None
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.model_plots_dict = {}
        self.num_outputs = num_outputs
        self.is_multi_task = num_outputs > 1
        self.task_kernel_type = task_kernel_type

    def setup_transformations(self, scaling_dict: dict):
        """
        Set up input and output transformations based on scaling configuration.
        
        Args:
            scaling_dict: Dictionary with 'input' and 'output' keys specifying scaling methods
            
        Returns:
            Tuple of (outcome_transform, input_transform) - can be None if no scaling
        """
        input_scaling_method = scaling_dict.get('input')
        output_scaling_method = scaling_dict.get('output')

        input_transformation_method = None
        outcome_transformation_method = None

        # Set up input transformation (currently only supports 'normalize')
        if input_scaling_method == 'normalize':
            input_transformation_method = Normalize(d=self.train_X.shape[1])
        elif input_scaling_method is None:
            pass
        else:
            raise ValueError("Invalid input scaling method. Expected 'normalize' or None.")

        # Set up output transformation (currently only supports 'standardize')
        if output_scaling_method == 'standardize':
            outcome_transformation_method = Standardize(m=self.train_Y.shape[1])
        elif output_scaling_method is None:
            pass
        else:
            raise ValueError("Invalid output scaling method. Expected 'standardize' or None.")

        return outcome_transformation_method, input_transformation_method   

    def setup_kernel_for_model(self):
        """
        Configure the kernel for single-task or multi-task scenarios.
        
        For multi-task models, this combines the input kernel with an appropriate
        task kernel (IndexKernel, ICM, etc.) to handle multiple outputs.
        """
        if self.is_multi_task:
            if self.task_kernel_type is None:
                # No task kernel - equivalent to separate SingleTaskGPs
                pass
            else:
                # Check if kernel is already a MultitaskKernel (from Hyperband)
                if hasattr(self.kernel, 'data_covar_module') and hasattr(self.kernel, 'task_covar_module'):
                    return
                
                # For multi-task models, create the appropriate task kernel
                if self.task_kernel_type == 'IndexKernel':
                    # Create task kernel and combine with input kernel
                    task_kernel = KernelFactory.create_task_kernel(
                        kernel_type=self.task_kernel_type, 
                        num_tasks=self.num_outputs,
                        base_kernel=self.kernel
                    )
                    self.kernel = KernelFactory.create_multi_task_kernel(self.kernel, task_kernel, num_tasks=self.num_outputs)
                elif self.task_kernel_type == 'ICM':
                    # ICM (Intrinsic Coregionalization Model) creates a combined kernel directly
                    self.kernel = KernelFactory.create_task_kernel(
                        kernel_type=self.task_kernel_type, 
                        num_tasks=self.num_outputs,
                        base_kernel=self.kernel
                    )
                elif self.task_kernel_type == 'LMC':
                    # LMC (Linear Model of Coregionalization) creates a combined kernel directly
                    self.kernel = KernelFactory.create_task_kernel(
                        kernel_type=self.task_kernel_type, 
                        num_tasks=self.num_outputs,
                        base_kernel=self.kernel
                    )
                else:
                    # Handle unknown task kernel types
                    raise ValueError(f"Unknown task kernel type: {self.task_kernel_type}. Expected 'IndexKernel', 'ICM', or 'LMC'")
        else:
            # For single-task models, use the input kernel as is
            pass

# TODO: option to not completely retrain and re-initiate the model? dict can be handed over? How long can the training time even be?

    def train(self, num_epochs: int, convergence_training: bool = True):
        """
        Train the GP model with the current training data.
        
        Args:
            num_epochs: Number of training epochs
            convergence_training: If True, use Adam optimizer with convergence training
                                 If False, use the specified optimizer_type
        """
        if convergence_training:
            # Setup transformations and kernel for both model types
            self.outcome_transform, self.input_transform = self.setup_transformations(scaling_dict=self.scaling_dict)
            self.setup_kernel_for_model()
            
            # Create model for both types
            self.gp_model = GPModelFactory.create_model(
                model_type=self.gp_model_type, 
                train_X=self.train_X, 
                train_Y=self.train_Y, 
                kernel=self.kernel, 
                likelihood=self.likelihood, 
                outcome_transform=self.outcome_transform, 
                input_transform=self.input_transform
            )
            
            # Unified training with Adam for both SingleTaskGP and MultiTaskGP
            print(f"Training {self.gp_model_type} model with Adam optimizer")
            
            # Initialize transforms with training data before training
            if self.outcome_transform is not None:
                self.outcome_transform.train()
                _ = self.outcome_transform(self.train_Y)
                self.outcome_transform.eval()
            
            if self.input_transform is not None:
                self.input_transform.train()
                _ = self.input_transform(self.train_X)
                self.input_transform.eval()
            
            # Setup optimizer and MLL
            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
            
            # Training loop
            self.gp_model.train()
            self.likelihood.train()
            
            for i in range(num_epochs):
                optimizer.zero_grad()
                output = self.gp_model(self.train_X)
                loss = -mll(output, self.train_Y)
                
                # Ensure loss is scalar for backward pass
                if loss.dim() > 0:
                    loss = loss.sum()
                
                loss.backward()
                
                if i % 20 == 0:
                    print(f'Iter {i+1}/{num_epochs} - Loss: {loss.item():.3f}')
                
                optimizer.step()
            
            print(f"Training completed. Final loss: {loss.item():.3f}")
            
        else:
            print(f"Performing training with {self.optimizer_type} optimizer, training over {num_epochs} epochs")
            from training import training
            self.gp_model = training.training_gp_model(gp_model=self.gp_model, optimizer=self.optimizer, mll=self.mll, train_X=self.train_X, train_Y=self.train_Y, num_epochs=num_epochs)

    def show_model_visualization(self):
        """Placeholder for model visualization functionality."""
        pass

    def add_point_to_dataset(self, new_X: torch.Tensor, new_Y: torch.Tensor):
        """
        Add new training points to the dataset.
        
        Args:
            new_X: New input points to add
            new_Y: New output points to add
            
        Raises:
            ValueError: If dimensions don't match the existing dataset
        """
        # Validate dimensions match existing dataset
        if new_X.shape[1] != self.train_X.shape[1]:
            raise ValueError(f"New input point has {new_X.shape[1]} dimensions, but model expects {self.train_X.shape[1]} dimensions")
        if new_Y.shape[1] != self.train_Y.shape[1]:
            raise ValueError(f"New output point has {new_Y.shape[1]} dimensions, but model expects {self.train_Y.shape[1]} dimensions")

        # Concatenate new points with existing dataset
        self.train_X = torch.cat([self.train_X, new_X], 0)
        self.train_Y = torch.cat([self.train_Y, new_Y], 0)

    def create_bounds_tensor(self, bounds_list):
        """
        Convert a list of bounds into a 2 x d tensor format for Gaussian Process models.

        Args:
            bounds_list: List of tuples, each containing (lower_bound, upper_bound) for a dimension

        Returns:
            torch.Tensor: A 2 x d tensor where the first row contains lower bounds 
                         and the second row contains upper bounds
        """
        # Extract lower and upper bounds into separate lists
        lower_bounds = [lb for lb, ub in bounds_list]
        upper_bounds = [ub for lb, ub in bounds_list]

        # Create 2 x d tensor with lower bounds in first row, upper bounds in second row
        return torch.tensor([lower_bounds, upper_bounds], dtype=torch.float64) 
