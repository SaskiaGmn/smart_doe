# Import only for SingleTaskGP (MultiTaskGP uses pure GPyTorch)
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import Likelihood
from gpytorch.kernels import Kernel
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.models import ExactGP
from typing import Optional
import torch
import gpytorch

class MultitaskGPModel(ExactGP):
    """
    Custom Multitask GP Model that uses MultitaskMultivariateNormal like in GPyTorch docs.
    """
    def __init__(self, train_x, train_y, likelihood, kernel, num_tasks, input_transform=None, outcome_transform=None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        self.covar_module = kernel
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform
        self.num_tasks = num_tasks

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultitaskMultivariateNormal(mean_x, covar_x)
    
    def transform_inputs(self, X):
        """Required by BoTorch's fit_gpytorch_mll"""
        return self.input_transform(X)
    
    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None):
        """
        Required by BoTorch's acquisition functions.
        Returns a posterior distribution for the given inputs.
        """
        posterior = self.forward(X)
        
        if self.outcome_transform is not None and hasattr(self.outcome_transform, 'means') and self.outcome_transform.means is not None:
            posterior = self.outcome_transform.untransform_posterior(posterior)
        
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        
        return posterior

class GPModelFactory:
    """
    Factory for creating GP models.
    Supports SingleTaskGP and custom MultitaskGPModel.
    """
    
    @staticmethod
    def create_model(model_type: str, train_X: torch.Tensor, train_Y: torch.Tensor, kernel: Kernel, likelihood: Likelihood, outcome_transform, input_transform, task_feature: Optional[int] = None):
        """
        Create a GP model based on the specified type.
        
        Args:
            model_type: Type of model ('SingleTaskGP', 'MultiTaskGP')
            train_X: Training input data
            train_Y: Training output data
            kernel: Kernel for the model
            likelihood: Likelihood for the model
            outcome_transform: Output transformation
            input_transform: Input transformation
            task_feature: Task feature index for MultiTaskGP
            
        Returns:
            GP model instance
        """
        if model_type == "SingleTaskGP":
            return SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                likelihood=likelihood,
                outcome_transform=outcome_transform,
                input_transform=input_transform
            )
        elif model_type == "MultiTaskGP":

            num_tasks = train_Y.shape[1]
            
            return MultitaskGPModel(
                train_x=train_X,
                train_y=train_Y,
                likelihood=likelihood,
                kernel=kernel,
                num_tasks=num_tasks,
                input_transform=input_transform,
                outcome_transform=outcome_transform
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")