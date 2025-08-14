from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound 
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from gpytorch.models import ExactGP
import torch

# TODO: is a transform in the acquisition function required? Excerpt from Expected Improvement:
'''
posterior_transform: A PosteriorTransform. If using a multi-output model,
a PosteriorTransform that transforms the multi-output posterior into a
single-output posterior is required.
'''

# TODO: Implementation of a parameter xi to adapt exploration/exploitation, can be implemented as a substraction from the mean (e.g. in EI) -> see https://github.com/pytorch/botorch/issues/373

class WeightedSumPosteriorTransform(PosteriorTransform):
    """
    Posterior transform that computes a weighted sum of multiple outputs.
    This transforms a multi-output posterior into a single-output posterior.
    """
    
    def __init__(self, weights: torch.Tensor):
        """
        Initialize the weighted sum posterior transform.
        
        Args:
            weights: Tensor of weights for each output dimension (should sum to 1)
        """
        super().__init__()
        self.weights = weights
        
        # Validate weights sum to 1
        if abs(weights.sum() - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def forward(self, posterior):
        """
        Transform the posterior to a weighted sum.
        
        Args:
            posterior: Multi-output posterior distribution
            
        Returns:
            Single-output posterior distribution
        """
        mean = posterior.mean  # Shape: [batch_size, num_outputs]
        variance = posterior.variance  # Shape: [batch_size, num_outputs]
        
        # Compute weighted sum of means and variances
        weighted_mean = (mean * self.weights).sum(dim=-1, keepdim=True)  # Shape: [batch_size, 1]
        weighted_variance = (variance * (self.weights ** 2)).sum(dim=-1, keepdim=True)  # Shape: [batch_size, 1]
        
        # Create new posterior with single output
        from gpytorch.distributions import MultivariateNormal
        return MultivariateNormal(weighted_mean.squeeze(-1), torch.diag_embed(weighted_variance.squeeze(-1)))
    
    def evaluate(self, Y):
        """
        Required abstract method for PosteriorTransform.
        
        Args:
            Y: Output tensor
            
        Returns:
            Transformed output tensor
        """
        return (Y * self.weights).sum(dim=-1, keepdim=True)

class WeightedSumAcquisitionFunction(AcquisitionFunction):
    """
    Custom acquisition function that computes a weighted sum of multiple outputs.
    This transforms a multi-output posterior into a single-output posterior.
    """
    
    def __init__(self, model: ExactGP, weights: torch.Tensor, maximize: bool = True):
        """
        Initialize the weighted sum acquisition function.
        
        Args:
            model: The GP model
            weights: Tensor of weights for each output dimension (should sum to 1)
            maximize: Whether to maximize or minimize
        """
        super().__init__(model)
        self.weights = weights
        self.maximize = maximize
        
        # Validate weights sum to 1
        if abs(weights.sum() - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
    
    def forward(self, X):
        """
        Compute the weighted sum acquisition function value.
        
        Args:
            X: Input tensor
            
        Returns:
            Weighted sum of the posterior means
        """
        posterior = self.model.posterior(X)
        mean = posterior.mean  # Shape: [batch_size, num_outputs]
        
        # Compute weighted sum and ensure 1D output
        weighted_sum = (mean * self.weights).sum(dim=-1)  # Shape: [batch_size]
        
        # Ensure output is 1D for BoTorch compatibility
        if weighted_sum.dim() > 1:
            weighted_sum = weighted_sum.squeeze()
        
        return weighted_sum

class AcquisitionFunctionFactory():

    @staticmethod
    def create_acquisition_function(acq_function_type: str, gp_model: ExactGP, train_Y: torch.Tensor, maximization: bool = True, **kwargs):

        """
        Factory method to create different types of acquisition functions used in Bayesian Optimization.

        Parameters:
        - acq_function_type (str): The type of acquisition function to create. Possible: 'Exp_Improvement', 'LogExp_Improvement', 'Prob_Improvement', 'Up_Conf_Bound', 'Weighted_Sum'
        - gp_model (ExactGP): Gaussian Process model used by the acquisition function.
        - train_Y (torch.Tensor): The outputs from the training data.
        - maximization (bool): Flag to indicate if the objective is maximization. Default is True.
        - kwargs: Additional keyword arguments for specific acquisition functions. 
                 For Weighted_Sum: 'weights' (torch.Tensor) - weights for each output dimension

        Returns:
        - An instance of the specified acquisition function.

        Raises:
        - ValueError: If an unsupported acquisition function type is provided.
        """
        
        if maximization:
            best_f = train_Y.max()
        else:
            best_f = train_Y.min()

        if acq_function_type == 'Exp_Improvement':
            return ExpectedImprovement(model = gp_model, best_f=best_f, maximize=maximization)
        elif acq_function_type == 'LogExp_Improvement':
            return LogExpectedImprovement(model=gp_model, best_f=best_f, maximize=maximization)
        elif acq_function_type == 'Prob_Improvement':
            return ProbabilityOfImprovement(model=gp_model, best_f=best_f, maximize=maximization)
        elif acq_function_type == 'Up_Conf_Bound':
            return UpperConfidenceBound(model=gp_model, maximize=maximization, **kwargs)
        elif acq_function_type == 'Weighted_Sum':
            weights = kwargs.get('weights', None)
            if weights is None:
                # Default to equal weights if not provided
                num_outputs = train_Y.shape[1] if train_Y.dim() > 1 else 1
                weights = torch.ones(num_outputs) / num_outputs
            
            # For multi-output models, use standard acquisition functions with posterior transform
            if train_Y.shape[1] > 1:
                posterior_transform = WeightedSumPosteriorTransform(weights)
                return LogExpectedImprovement(
                    model=gp_model, 
                    best_f=best_f, 
                    maximize=maximization,
                    posterior_transform=posterior_transform
                )
            else:
                return WeightedSumAcquisitionFunction(model=gp_model, weights=weights, maximize=maximization)
        else:
            raise ValueError(f"Unsupported acquisition function type: {acq_function_type}")