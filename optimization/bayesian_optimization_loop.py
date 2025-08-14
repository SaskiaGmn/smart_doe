import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from optimization.acquisition_function_factory import AcquisitionFunctionFactory

class BayesianOptimizationLoop:
    def __init__(self, base_model, acq_func_type: str = "LogExp_Improvement", is_maximization: bool = True, acq_func_kwargs: dict = None, optimization_directions: list = None):
        self.base_model = base_model
        self.acq_func_type = acq_func_type
        self.is_maximization = is_maximization
        self.acq_func = None
        self.acq_func_values = []
        self.best_value = float('-inf') if is_maximization else float('inf')
        self.best_parameters = None
        self.acq_func_kwargs = acq_func_kwargs or {}
        self.optimization_directions = optimization_directions or ['maximize']

        if 'weights' in self.acq_func_kwargs:
            self.weights = self.acq_func_kwargs['weights']
        else:
            self.weights = None
            
    def update_model(self, input_value, original_x):
        """Updates the model with a new observation.
        Args:   
            input_value: A single value (float) for single-output, or a Tensor with multiple values for multi-output
            original_x: A Tensor with the input parameters
        """
        # Ensure original_x has batch dimension (2D tensor)
        if original_x.dim() == 1:
            original_x = original_x.unsqueeze(0)
            
        # Ensure input_value is a 2D tensor for consistent processing
        if not isinstance(input_value, torch.Tensor):
            input_value = torch.tensor([[input_value]], dtype=torch.float32)
        elif input_value.dim() == 1:
            input_value = input_value.unsqueeze(0)

        transformed_input_value = self.apply_optimization_directions(input_value)
            
        # Update best value and parameters if necessary
        if self.base_model.is_multi_task:
            # For multi-output, compute weighted sum for best value tracking
            if hasattr(self, 'weights') and self.weights is not None:
                current_value = (transformed_input_value * self.weights).sum().item()
            else:
                # Sum of all outputs (default behavior for multi-task)
                current_value = transformed_input_value.sum().item()
        else:
            # Single-output
            current_value = transformed_input_value.item()
            
        if (self.is_maximization and current_value > self.best_value) or \
           (not self.is_maximization and current_value < self.best_value):
            self.best_value = current_value
            self.best_parameters = original_x.clone()
            
        # Add the new point to the dataset (use transformed values for training)
        self.base_model.add_point_to_dataset(new_X=original_x, new_Y=transformed_input_value)
        self.base_model.train(num_epochs=100)
        
    def apply_optimization_directions(self, input_value):
        """Applies optimization direction transformations to input values.
        
        This method handles the conversion between minimization and maximization problems
        by negating values when minimization is specified. For multi-task problems,
        each output can have its own optimization direction.
        
        Args:
            input_value: Tensor with output values
            
        Returns:
            Tensor with transformed values (negative for minimization)
        """
        if not self.base_model.is_multi_task:
            # Single output: apply the single optimization direction
            if self.optimization_directions[0] == 'minimize':
                return -input_value
            else:
                return input_value
        else:
            # Multi-output: apply direction for each output
            transformed = input_value.clone()
            for i, direction in enumerate(self.optimization_directions):
                if i < input_value.shape[1] and direction == 'minimize':
                    transformed[:, i] = -transformed[:, i]
            return transformed
            
    def optimization_iteration(self, num_restarts=40, raw_samples=400):
        """Performs an optimization step.
        
        Args:
            num_restarts: Number of random restarts for optimization
            raw_samples: Number of raw samples to evaluate
            
        Returns:
            candidate: Tensor with the proposed input parameters
            acq_value: The value of the acquisition function
        """
        try:
            # Create the acquisition function using the factory
            self.acq_func = AcquisitionFunctionFactory.create_acquisition_function(
                acq_function_type=self.acq_func_type,
                gp_model=self.base_model.gp_model,
                train_Y=self.base_model.train_Y,
                maximization=self.is_maximization,
                **self.acq_func_kwargs
            )
            
            bounds = self.base_model.bounds_list.detach().clone().to(torch.float32)
            
            candidate, acq_value = optimize_acqf(
                acq_function=self.acq_func,  # The acquisition function to optimize
                bounds=bounds,  # Bounds for each input dimension [lower_bounds, upper_bounds]
                q=1,  # Number of points to acquire (batch size) - we acquire one point at a time
                num_restarts=num_restarts,  # Number of random restarts for the optimizer to avoid local optima
                raw_samples=raw_samples,  # Number of random samples to evaluate before optimization
            )
            
            self.acq_func_values.append(acq_value.item())
            return candidate, acq_value
        except Exception as e:
            print(f"Error in optimization_iteration: {str(e)}")
            raise
            
    def get_optimization_status(self):
        """Returns the current status of the optimization.
        
        Returns:
            dict: Dictionary containing optimization status information
        """
        best_value = self.best_value
        if best_value == float('inf'):
            best_value = 1e10
        elif best_value == float('-inf'):
            best_value = -1e10
            
        best_params = None
        if self.best_parameters is not None:
            best_params = self.best_parameters.tolist()
            
        acq_values = [float(v) for v in self.acq_func_values]
        
        return {
            'best_value': float(best_value),
            'best_parameters': best_params,
            'num_iterations': len(self.acq_func_values),
            'acquisition_values': acq_values
        } 