from typing import Dict, List, Any, Tuple, Optional
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import MarginalLogLikelihood
from models.gp_model import BaseGPModel
from models.likelihood_factory import LikelihoodFactory
from models.mll_factory import MLLFactory
from kernel_choice.kernel_candidates import KernelCandidates
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class KernelEvaluationResult:
    """Data class to store kernel evaluation results."""
    kernel_name: str
    kernel_type: str
    kernel_instance: Kernel
    mll_score: float
    loo_cv_score: float
    weighted_score: float
    converged: bool = True

class KernelEvaluator:
    """
    Evaluates kernels using MLL and Leave-One-Out Cross-Validation.
    Minimal implementation for optimal kernel selection.
    """
    
    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor, 
                 bounds_list: List[Tuple], scaling_dict: Optional[Dict] = None,
                 mll_weight: float = 0.7, cv_weight: float = 0.3):
        """
        Initialize the kernel evaluator.
        
        Args:
            train_X: Training input data
            train_Y: Training output data
            bounds_list: Bounds for the input dimensions
            scaling_dict: Scaling configuration for input/output
            mll_weight: Weight for Marginal Log-Likelihood (default: 0.7)
            cv_weight: Weight for Cross-Validation (default: 0.3)
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.bounds_list = bounds_list
        self.scaling_dict = scaling_dict or {'input': 'normalize', 'output': 'standardize'}
        self.likelihood = LikelihoodFactory.create_likelihood('Gaussian')
        self.mll_weight = mll_weight
        self.cv_weight = cv_weight
        
        # Validate weights
        if abs(mll_weight + cv_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
    def evaluate_kernel(self, kernel_config: Dict[str, Any], 
                       training_epochs: int = 100) -> KernelEvaluationResult:
        """
        Evaluates a single kernel configuration.
        
        Args:
            kernel_config: Kernel configuration dictionary
            training_epochs: Number of training epochs
            
        Returns:
            KernelEvaluationResult with evaluation metrics
        """
        try:
            # Create kernel instance
            kernel = KernelCandidates.create_kernel_from_config(kernel_config)
            
            # Create and train GP model
            gp_model = BaseGPModel(
                model_type="SingleTaskGP",
                mll_type="ExactMarginalLogLikelihood",
                optimizer_type="adam",
                kernel=kernel,
                train_X=self.train_X,
                train_Y=self.train_Y,
                likelihood=self.likelihood,
                bounds_list=self.bounds_list,
                scaling_dict=self.scaling_dict
            )
            
            # Train the model
            gp_model.train(num_epochs=training_epochs, convergence_training=True)
            
            # Compute metrics
            mll_score = self._compute_marginal_log_likelihood(gp_model)
            loo_cv_score = self._compute_leave_one_out_cv(gp_model)
            
            # Compute weighted score
            weighted_score = (self.mll_weight * mll_score + 
                            self.cv_weight * loo_cv_score)
            
            return KernelEvaluationResult(
                kernel_name=kernel_config['name'],
                kernel_type=kernel_config['type'],
                kernel_instance=kernel,
                mll_score=mll_score,
                loo_cv_score=loo_cv_score,
                weighted_score=weighted_score
            )
            
        except Exception as e:
            return KernelEvaluationResult(
                kernel_name=kernel_config['name'],
                kernel_type=kernel_config['type'],
                kernel_instance=None,
                mll_score=float('-inf'),
                loo_cv_score=float('-inf'),
                weighted_score=float('-inf'),
                converged=False
            )
    
    def evaluate_kernels(self, kernel_configs: List[Dict[str, Any]], 
                        training_epochs: int = 100) -> List[KernelEvaluationResult]:
        """
        Evaluates multiple kernel configurations.
        
        Args:
            kernel_configs: List of kernel configuration dictionaries
            training_epochs: Number of training epochs for each kernel
            
        Returns:
            List of KernelEvaluationResult objects
        """
        results = []
        
        for config in kernel_configs:
            result = self.evaluate_kernel(config, training_epochs)
            results.append(result)
        
        return results
    
    def _compute_marginal_log_likelihood(self, gp_model: BaseGPModel) -> float:
        """Compute the marginal log-likelihood."""
        try:
            mll = MLLFactory.create_mll('ExactMarginalLogLikelihood', gp_model.gp_model, gp_model.likelihood)
            gp_model.gp_model.eval()
            with torch.no_grad():
                output = gp_model.gp_model(gp_model.train_X)
                mll_value = mll(output, gp_model.train_Y).item()
            return mll_value
        except:
            return float('-inf')
    
    def _compute_leave_one_out_cv(self, gp_model: BaseGPModel) -> float:
        """
        Compute Leave-One-Out Cross-Validation score.
        Uses the predictive log-likelihood for each left-out point.
        """
        try:
            gp_model.gp_model.eval()
            cv_scores = []
            
            with torch.no_grad():
                for i in range(len(self.train_X)):
                    # Create training data without point i
                    train_X_loo = torch.cat([self.train_X[:i], self.train_X[i+1:]])
                    train_Y_loo = torch.cat([self.train_Y[:i], self.train_Y[i+1:]])
                    
                    # Create temporary model for LOO
                    temp_model = BaseGPModel(
                        model_type="SingleTaskGP",
                        mll_type="ExactMarginalLogLikelihood",
                        optimizer_type="adam",
                        kernel=gp_model.kernel,
                        train_X=train_X_loo,
                        train_Y=train_Y_loo,
                        likelihood=self.likelihood,
                        bounds_list=self.bounds_list,
                        scaling_dict=self.scaling_dict
                    )
                    
                    # Quick training (fewer epochs for LOO)
                    temp_model.train(num_epochs=20, convergence_training=True)
                    
                    # Predict for left-out point
                    test_X = self.train_X[i:i+1]
                    test_Y = self.train_Y[i:i+1]
                    
                    posterior = temp_model.gp_model.posterior(test_X)
                    mean = posterior.mean
                    variance = posterior.variance
                    
                    # Compute log-likelihood for left-out point
                    log_likelihood = self._compute_point_log_likelihood(test_Y, mean, variance)
                    cv_scores.append(log_likelihood)
            
            return np.mean(cv_scores)
            
        except Exception as e:
            return float('-inf')
    
    def _compute_point_log_likelihood(self, y_true: torch.Tensor, 
                                    y_pred_mean: torch.Tensor, 
                                    y_pred_var: torch.Tensor) -> float:
        """Compute log-likelihood for a single point."""
        try:
            # Add small constant to avoid numerical issues
            y_pred_var = y_pred_var + 1e-8
            
            # Compute log-likelihood
            log_likelihood = -0.5 * (
                torch.log(2 * np.pi * y_pred_var) + 
                (y_true - y_pred_mean)**2 / y_pred_var
            )
            
            return log_likelihood.item()
        except:
            return float('-inf')
    
    def get_best_kernel(self, results: List[KernelEvaluationResult]) -> Tuple[KernelEvaluationResult, Kernel]:
        """
        Selects the best kernel based on the weighted score.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Tuple of (best_result, best_kernel)
        """
        # Filter out failed evaluations
        valid_results = [r for r in results if r.converged]
        
        if not valid_results:
            raise ValueError("No kernels converged successfully!")
        
        # Select best based on weighted score
        best_result = max(valid_results, key=lambda x: x.weighted_score)
        
        return best_result, best_result.kernel_instance 