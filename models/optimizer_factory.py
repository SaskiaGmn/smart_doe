from torch.optim import Adam
from typing import Union, Iterable
import torch
from torch.optim.optimizer import ParamsT


class OptimizerFactory:
    """
    Factory class for creating PyTorch optimizers.
    
    This factory supports the configuration and creation of various types of optimizers based on input specifications,
    allowing for flexible and dynamic instantiation of optimizer objects with custom settings.
    
    Currently supports:
        - Adam: Adaptive Moment Estimation optimizer that combines the benefits of AdaGrad and RMSprop
    """

    @staticmethod
    def create_optimizer(type: str, model_parameters: ParamsT, **kwargs) -> torch.optim.Optimizer:
        """
        Creates and returns an optimizer based on the specified type and associated parameters.
        
        Adam Optimizer:
        Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that's been designed 
        to handle sparse gradients on noisy problems. It computes adaptive learning rates for each parameter by 
        estimating first and second moments of the gradients. Adam combines the advantages of two other extensions 
        of stochastic gradient descent: AdaGrad and RMSprop.
        
        Key features:
        - Adaptive learning rates for each parameter
        - Bias correction for first and second moment estimates
        - Works well with sparse gradients
        - Generally requires little tuning of hyperparameters
        
        Parameters:
            type (str): Type of optimizer to create. Currently only supports 'adam'.
            model_parameters (ParamsT): Parameters of the model that the optimizer will update.
            **kwargs: Additional keyword arguments for the optimizer:
                - lr (float): Learning rate (default: 0.001)
                - betas (tuple): Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
                - eps (float): Term added to denominator to improve numerical stability (default: 1e-8)
                - weight_decay (float): Weight decay (L2 penalty) (default: 0)
                
        Returns:
            torch.optim.Optimizer: An instance of a PyTorch optimizer configured according to the specified type
                                   and parameters.
                                   
        Raises:
            ValueError: If an unsupported optimizer type is specified.
        """
        if type == 'adam':
            return Adam(model_parameters, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {type}. Currently only 'adam' is supported.")