import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from optimization.acquisition_function_factory import AcquisitionFunctionFactory

class BayesianOptimizationLoop:
    def __init__(self, base_model, acq_func_type: str = "LogExp_Improvement", is_maximization: bool = True):
        self.base_model = base_model
        self.acq_func_type = acq_func_type
        self.is_maximization = is_maximization
        self.acq_func = None
        self.acq_func_values = []
        self.best_value = float('-inf') if is_maximization else float('inf')
        self.best_parameters = None
        
    def update_model(self, input_value, original_x):
        """Aktualisiert das Modell mit einer neuen Beobachtung.
        
        Args:
            input_value: Ein einzelner Wert (float) - der Output
            original_x: Ein Tensor mit den Input-Parametern
        """
        # Ensure that original_x has the correct shape (2D)
        if original_x.dim() == 1:
            original_x = original_x.unsqueeze(0)  # [n] -> [1, n]
            
        # Convert input_value to a 2D Tensor
        if not isinstance(input_value, torch.Tensor):
            input_value = torch.tensor([[input_value]], dtype=torch.float32)  # [[y]]
        elif input_value.dim() == 1:
            input_value = input_value.unsqueeze(0)  # [y] -> [[y]]
            
        # Update best value and parameters if necessary
        current_value = input_value.item()
        if (self.is_maximization and current_value > self.best_value) or \
           (not self.is_maximization and current_value < self.best_value):
            self.best_value = current_value
            self.best_parameters = original_x.clone()
            
        # Add the new point to the dataset
        self.base_model.add_point_to_dataset(new_X=original_x, new_Y=input_value)
        self.base_model.train(num_epochs=100)
        
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
                maximization=self.is_maximization
            )
            
            # Convert the bounds to the correct format
            bounds = torch.tensor(self.base_model.bounds_list, dtype=torch.float32)
            
            # Optimize the acquisition function
            candidate, acq_value = optimize_acqf(
                acq_function=self.acq_func,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
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
        # Convert best_value to a regular float if it's inf/-inf
        best_value = self.best_value
        if best_value == float('inf'):
            best_value = 1e10
        elif best_value == float('-inf'):
            best_value = -1e10
            
        # Convert best_parameters to list or None
        best_params = None
        if self.best_parameters is not None:
            best_params = self.best_parameters.tolist()
            
        # Ensure acquisition_values are all floats
        acq_values = [float(v) for v in self.acq_func_values]
        
        return {
            'best_value': float(best_value),
            'best_parameters': best_params,
            'num_iterations': len(self.acq_func_values),
            'acquisition_values': acq_values
        } 