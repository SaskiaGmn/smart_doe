from typing import Union
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

class MLLFactory:
    @staticmethod
    def create_mll(type: str, model: Union[SingleTaskGP], likelihood: Likelihood) -> ExactMarginalLogLikelihood:
        """
        Creates and returns a Marginal Log Likelihood (MLL) object for Gaussian Process model training.
        
        The Marginal Log Likelihood is a key objective function in GP training that measures how well
        the model fits the training data while accounting for the uncertainty in the GP predictions.
        It balances data fit (likelihood) with model complexity (prior), helping to prevent overfitting.
        
        The MLL is computed as: log p(y|X) = log ∫ p(y|f,X) p(f|X) df
        where p(y|f,X) is the likelihood and p(f|X) is the GP prior.
        
        Parameters:
            type (str): Type of MLL to create. Currently only supports 'ExactMarginalLogLikelihood'.
            model (SingleTaskGP): The GP model for which the MLL is computed.
            likelihood (Likelihood): The likelihood function (e.g., GaussianLikelihood) associated with the GP model.
            
        Returns:
            ExactMarginalLogLikelihood: An instance of Marginal Log Likelihood for training the GP model.
            
        Raises:
            ValueError: If an unsupported MLL type is specified.
        """
        if type == 'ExactMarginalLogLikelihood':
            return ExactMarginalLogLikelihood(likelihood, model)
        else:
            raise ValueError(f"Unsupported MLL type: {type}. Currently only 'ExactMarginalLogLikelihood' is supported.")