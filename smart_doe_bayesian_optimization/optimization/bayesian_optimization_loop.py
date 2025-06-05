import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

class BayesianOptimizationLoop:
    def __init__(self, base_model):
        self.base_model = base_model
        self.acq_func_values = []
        
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
            
        # Add the new point to the dataset
        self.base_model.add_point_to_dataset(new_X=original_x, new_Y=input_value)
        self.base_model.train(num_epochs=100)
        
    def optimization_iteration(self, num_restarts=40, raw_samples=400):
        """Performs an optimization step.
        
        Returns:
            candidate: Tensor with the proposed input parameters
            acq_value: The value of the acquisition function
        """
        try:
            # Create the acquisition function
            acq_func = LogExpectedImprovement(
                model=self.base_model.gp_model,
                best_f=self.base_model.train_Y.max().item()
            )
            
            # Convert the bounds to the correct format
            bounds = torch.tensor(self.base_model.bounds_list, dtype=torch.float32)
            
            # Optimize the acquisition function
            candidate, acq_value = optimize_acqf(
                acq_function=acq_func,
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