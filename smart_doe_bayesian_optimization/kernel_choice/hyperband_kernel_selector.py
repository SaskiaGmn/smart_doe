from typing import Dict, List, Any, Tuple, Optional
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from models.gp_model import BaseGPModel
from models.likelihood_factory import LikelihoodFactory
from models.mll_factory import MLLFactory
from models.kernel_factory import KernelFactory
from models.gp_model_factory import GPModelFactory
import torch
import numpy as np
import time
import random
from dataclasses import dataclass
from sampler.lhs import build_lhs
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from sklearn.model_selection import KFold

@dataclass
class HyperbandConfig:
    """Configuration for Hyperband optimization."""
    max_iter: int
    eta: int
    epochs_per_level: List[int]
    max_time_seconds: int

class HyperbandKernelSelector:
    """
    Hyperband-based kernel selection with adaptive resource allocation.
    Optimizes both kernel type and hyperparameters simultaneously.
    """
    
    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor, 
                 bounds_list: List[Tuple], scaling_dict: Optional[Dict] = None,
                 mll_weight: float = 0.7, cv_weight: float = 0.3):
        """
        Initialize the Hyperband kernel selector.
        
        Args:
            train_X: Training input data
            train_Y: Training output data
            bounds_list: Bounds for the input dimensions
            scaling_dict: Scaling configuration for input/output
            mll_weight: Weight for Marginal Log-Likelihood 
            cv_weight: Weight for Cross-Validation 
        """
        self.train_X = train_X
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        self.train_Y = train_Y
        self.bounds_list = bounds_list
        self.scaling_dict = scaling_dict or {}
        self.mll_weight = mll_weight
        self.cv_weight = cv_weight
        
        # Validate weights
        if abs(mll_weight + cv_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        # Initialize likelihood
        self.likelihood = LikelihoodFactory.create_likelihood('Gaussian')
        
        # Determine configuration based on dataset size
        self.config = self.get_adaptive_config()
        
        # Pre-generate LHS samples for better coverage
        self.generate_lhs_samples()
        
        self.sample_indices = {kernel_type: 0 for kernel_type in self.lhs_samples.keys()}
        
    def get_adaptive_config(self) -> HyperbandConfig:
        """
        Get adaptive configuration based on dataset size.
        
        Returns:
            HyperbandConfig with appropriate parameters
        """
        n_samples = len(self.train_X)
        
        if n_samples < 50:
            # Small datasets: fewer resources
            return HyperbandConfig(
                max_iter=50,
                eta=2,
                epochs_per_level=[10, 25, 50],
                max_time_seconds=300  # 5 minutes
            )
        else:
            # Large datasets: more resources
            return HyperbandConfig(
                max_iter=100,
                eta=3,
                epochs_per_level=[10, 25, 50, 100],
                max_time_seconds=600  # 10 minutes
            )
    
    def select_kernel(self, random_seed: Optional[int] = None) -> Kernel:
        """
        Select the best kernel using Hyperband optimization.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Best kernel instance
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        # Run Hyperband optimization
        best_config = self.run_hyperband_optimization()
        
        # Create and return the best kernel
        return self.create_kernel_from_config(best_config)
    
    def run_hyperband_optimization(self) -> Dict[str, Any]:
        """
        Run Hyperband optimization to find the best kernel configuration.
        
        Returns:
            Best kernel configuration dictionary
        """
        start_time = time.time()
        best_score = float('-inf')
        best_config = None
        
        # Calculate number of brackets
        s_max = len(self.config.epochs_per_level) - 1
        
        for s in range(s_max, -1, -1):
            # Number of configurations to try in this bracket
            n = int(np.ceil(self.config.max_iter / (s + 1) * (self.config.eta ** s)))
            
            # Number of configurations to keep after each round
            r = self.config.epochs_per_level[0] * (self.config.eta ** s)
            
            # Generate random configurations
            configs = [self.generate_random_kernel_config() for _ in range(n)]
            
            # Successive halving
            for i in range(s + 1):
                # Number of configurations to evaluate in this round
                n_i = int(np.ceil(n / (self.config.eta ** i)))
                # Number of epochs for this round
                r_i = int(r * (self.config.eta ** i))
                
                # Evaluate configurations
                scores = []
                for j, config in enumerate(configs[:n_i]):
                    # Check time limit
                    if time.time() - start_time > self.config.max_time_seconds:
                        break
                    
                    score = self.evaluate_kernel_config(config, r_i)
                    scores.append(score)
                    
                    # Update best if better
                    if score > best_score:
                        best_score = score
                        best_config = config.copy()
                
                # Early stopping if no improvement
                if len(scores) > 0 and max(scores) <= best_score:
                    break
                
                # Keep top configurations for next round
                if i < s:
                    # Sort by score and keep top 1/eta
                    config_scores = list(zip(configs[:n_i], scores))
                    config_scores.sort(key=lambda x: x[1], reverse=True)
                    configs = [config for config, _ in config_scores[:n_i // self.config.eta]]
        
        if best_config is None:
            raise ValueError("No valid kernel configuration found!")
        
        return best_config
    
    def generate_random_kernel_config(self) -> Dict[str, Any]:
        """
        Generate a kernel configuration using pre-generated LHS samples.
        Ensures equal distribution across kernel types.
        
        Returns:
            Kernel configuration dictionary
        """
        kernel_types = ['RBF', 'Matern', 'Linear', 'Periodic', 'Polynomial']
        
        # Calculate how many configurations we need per kernel type
        total_configs_needed = self.config.max_iter
        configs_per_kernel = total_configs_needed // len(kernel_types)
        
        # Determine which kernel type to generate based on current count
        current_total = sum(self.sample_indices.values())
        kernel_type_idx = current_total // configs_per_kernel
        
        # If we've generated enough of each type, cycle through them
        if kernel_type_idx >= len(kernel_types):
            kernel_type_idx = current_total % len(kernel_types)
        
        kernel_type = kernel_types[kernel_type_idx]
        
        config = {'type': kernel_type}
        
        # Get next LHS sample for this kernel type
        sample_idx = self.sample_indices[kernel_type]
        sample = self.lhs_samples[kernel_type][sample_idx]
        
        # Convert tensor to list for easier handling
        sample_list = sample.tolist() if hasattr(sample, 'tolist') else list(sample)
        
        if kernel_type == 'RBF':
            config['lengthscale'] = sample_list[0]
            config['variance'] = sample_list[1]
        elif kernel_type == 'Matern':
            config['lengthscale'] = sample_list[0]
            config['nu'] = sample_list[1]
            config['variance'] = sample_list[2]
        elif kernel_type == 'Linear':
            config['variance'] = sample_list[0]
            config['offset'] = sample_list[1]
        elif kernel_type == 'Periodic':
            config['lengthscale'] = sample_list[0]
            config['period'] = sample_list[1]
            config['variance'] = sample_list[2]
        elif kernel_type == 'Polynomial':
            config['variance'] = sample_list[0]
            config['offset'] = sample_list[1]
            config['power'] = sample_list[2]
        
        # Update sample index
        self.sample_indices[kernel_type] = (self.sample_indices[kernel_type] + 1) % len(self.lhs_samples[kernel_type])
        
        return config
    
    def evaluate_kernel_config(self, config: Dict[str, Any], epochs: int) -> float:
        """
        Evaluate a kernel configuration by training a GP model and computing scores.
        
        Args:
            config: Kernel configuration dictionary
            epochs: Number of training epochs
            
        Returns:
            Combined score (weighted MLL + LOO-CV)
        """
        try:
            # Create kernel
            kernel = self.create_kernel_from_config(config)
            
            # Create transforms
            outcome_transform = Standardize(m=self.train_Y.shape[1])
            input_transform = Normalize(d=self.train_X.shape[1])
            
            # Create GP model
            gp_model = GPModelFactory.create_model(
                model_type="SingleTaskGP",
                train_X=self.train_X,
                train_Y=self.train_Y,
                kernel=kernel,
                likelihood=self.likelihood,
                outcome_transform=outcome_transform,
                input_transform=input_transform
            )
            
            # Train model
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            trained_epochs = self.train_with_early_stopping(gp_model, mll, epochs)
            
            # Compute MLL score
            with torch.no_grad():
                output = gp_model(gp_model.train_inputs[0])
                mll_score = -mll(output, gp_model.train_targets).item()
                
                # Check if score is valid
                if torch.isnan(torch.tensor(mll_score)) or torch.isinf(torch.tensor(mll_score)):
                    return float('-inf')
                
                # Compute CV score if weight > 0
                if self.cv_weight > 0:
                    cv_score = self.compute_kfold_cv_score(gp_model, mll)
                    
                    # Simple normalization: ensure both scores are positive and comparable
                    # MLL scores are typically negative, CV scores are typically negative
                    # Make both positive and scale to similar ranges
                    mll_positive = abs(mll_score)  # Convert to positive
                    cv_positive = abs(cv_score)    # Convert to positive
                    
                    # Scale to [0, 1] range (assuming max reasonable values)
                    mll_normalized = min(1.0, mll_positive / 50.0)  # Max MLL around 50
                    cv_normalized = min(1.0, cv_positive / 25.0)    # Max CV around 25
                    
                    combined_score = self.mll_weight * mll_normalized + self.cv_weight * cv_normalized
                else:
                    combined_score = mll_score
                
                return combined_score
                
        except Exception as e:
            return float('-inf')
    
    def setup_transformations(self):
        """Setup transformations like in BaseGPModel."""
        input_scaling_method = self.scaling_dict.get('input') if self.scaling_dict else None
        output_scaling_method = self.scaling_dict.get('output') if self.scaling_dict else None

        input_transformation_method = None
        outcome_transformation_method = None

        if input_scaling_method == 'normalize':
            input_transformation_method = Normalize(d=self.train_X.shape[1])
        elif input_scaling_method is None:
            pass
        else:
            raise ValueError("Invalid input scaling method. Expected 'normalize' or None.")

        if output_scaling_method == 'standardize':
            outcome_transformation_method = Standardize(m=self.train_Y.shape[1])
        elif output_scaling_method is None:
            pass
        else:
            raise ValueError("Invalid output scaling method. Expected 'standardize' or None.")

        return outcome_transformation_method, input_transformation_method
    
    def train_with_early_stopping(self, gp_model: BaseGPModel, mll: ExactMarginalLogLikelihood, max_epochs: int) -> int:
        """
        Train GP model with early stopping based on convergence detection.
        
        Args:
            gp_model: GP model to train (SingleTaskGP object)
            mll: Marginal Log-Likelihood for the model
            max_epochs: Maximum number of epochs
            
        Returns:
            Number of epochs actually trained
        """
        # Create optimizer
        from models.optimizer_factory import OptimizerFactory
        optimizer = OptimizerFactory.create_optimizer(
            type="adam",
            model_parameters=gp_model.parameters(),
            lr=0.1
        )
        
        # Set model to training mode
        gp_model.train()
        
        # Early stopping parameters
        patience = 10  # Number of epochs to wait for improvement
        min_improvement = 1e-4  # Minimum improvement threshold
        best_loss = float('inf')
        patience_counter = 0
        
        # Store loss history for convergence detection
        loss_history = []
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            output = gp_model(gp_model.train_inputs[0])
            loss = -mll(output, gp_model.train_targets)
            # Ensure loss is scalar for backward pass
            if loss.dim() > 0:
                loss = loss.sum()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Check for improvement
            if current_loss < best_loss - min_improvement:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping conditions
            if patience_counter >= patience:
                # Check if we have enough epochs for meaningful training
                if epoch >= 20:  # Minimum 20 epochs
                    return epoch + 1
            
            # Additional convergence check: loss stability
            if len(loss_history) >= 15:
                recent_losses = loss_history[-15:]
                loss_std = np.std(recent_losses)
                if loss_std < 1e-5:  # Very stable loss
                    return epoch + 1
        
        return max_epochs
    
    def compute_marginal_log_likelihood(self, gp_model: BaseGPModel) -> float:
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
    
    def compute_kfold_cv_score(self, gp_model, mll):
        """
        Compute K-Fold Cross-Validation score for the GP model using sklearn's KFold.
        
        Args:
            gp_model: Trained GP model (SingleTaskGP object)
            mll: Marginal Log-Likelihood for the model
            
        Returns:
            Average K-Fold CV score
        """
        X = gp_model.train_inputs[0]
        Y = gp_model.train_targets
        n = X.shape[0]
        
        # Use sklearn's KFold for robust splitting
        k_folds = min(5, n // 2) if n >= 6 else 3
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        scores = []
        
        # Use sklearn's KFold for splitting, but keep GP-specific evaluation
        for train_idx, test_idx in kfold.split(X):
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]
            
            try:
                # Create new model for this fold
                model = GPModelFactory.create_model(
                    model_type="SingleTaskGP",
                    train_X=X_train,
                    train_Y=Y_train,
                    kernel=gp_model.covar_module,
                    likelihood=gp_model.likelihood,
                    outcome_transform=gp_model.outcome_transform,
                    input_transform=gp_model.input_transform
                )
                
                # Quick training for CV (fewer epochs for speed)
                model.train()
                mll_cv = ExactMarginalLogLikelihood(model.likelihood, model)
                
                # Train CV model with same approach as main model
                optimizer_cv = torch.optim.Adam(model.parameters(), lr=0.1)
                
                # Use same training approach as main model but with early stopping
                patience = 5  # Shorter patience for CV
                best_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(min(50, len(X_train) * 2)):  # More reasonable max epochs
                    optimizer_cv.zero_grad()
                    output = model(X_train)
                    loss = -mll_cv(output, Y_train)
                    if loss.dim() > 0:
                        loss = loss.sum()
                    loss.backward()
                    optimizer_cv.step()
                    
                    current_loss = loss.item()
                    
                    # Early stopping for CV
                    if current_loss < best_loss - 1e-4:
                        best_loss = current_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience and epoch >= 10:  # Minimum 10 epochs
                        break
                
                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    output = model(X_test)
                    score = -mll_cv(output, Y_test).item()  # Make positive for consistency
                scores.append(score)
                
            except Exception as e:
                # Error in this fold: count as -inf
                scores.append(float('-inf'))
        
        # Return average score, or -inf if all failed
        return sum(scores) / len(scores) if scores else float('-inf')
    
    def create_kernel_from_config(self, config: Dict[str, Any]) -> Kernel:
        """
        Create a kernel instance from configuration.
        
        Args:
            config: Kernel configuration dictionary
            
        Returns:
            Kernel instance
        """
        try:
            kernel_type = config['type']
            
            if kernel_type == 'RBF':
                # Convert tensor parameters to floats
                lengthscale = float(config['lengthscale'])
                variance = float(config['variance'])
                
                # RBF doesn't have variance parameter, wrap with ScaleKernel
                base_kernel = KernelFactory.create_kernel('RBF', lengthscale=lengthscale)
                return ScaleKernel(base_kernel, outputscale=variance)
                
            elif kernel_type == 'Matern':
                # Convert tensor parameters to floats
                lengthscale = float(config['lengthscale'])
                nu_raw = float(config['nu'])
                variance = float(config['variance'])
                
                # Matern kernel only accepts nu values: 0.5, 1.5, 2.5
                # Round to nearest valid value
                valid_nu_values = [0.5, 1.5, 2.5]
                nu = min(valid_nu_values, key=lambda x: abs(x - nu_raw))
                
                # Matern doesn't have variance parameter, wrap with ScaleKernel
                base_kernel = KernelFactory.create_kernel('Matern', lengthscale=lengthscale, nu=nu)
                return ScaleKernel(base_kernel, outputscale=variance)
                
            elif kernel_type == 'Linear':
                # Convert tensor parameters to floats
                variance = float(config['variance'])
                offset = float(config['offset'])
                
                # Linear has variance parameter, no need for ScaleKernel
                return KernelFactory.create_kernel('Linear', variance=variance, offset=offset)
                
            elif kernel_type == 'Periodic':
                # Convert tensor parameters to floats
                lengthscale = float(config['lengthscale'])
                period = float(config['period'])
                variance = float(config['variance'])
                
                # Periodic doesn't have variance parameter, wrap with ScaleKernel
                base_kernel = KernelFactory.create_kernel('Periodic', lengthscale=lengthscale, period=period)
                return ScaleKernel(base_kernel, outputscale=variance)
                
            elif kernel_type == 'Polynomial':
                # Convert tensor parameters to floats
                variance = float(config['variance'])
                offset = float(config['offset'])
                power = int(config['power'])  # Power should be integer
                
                # Polynomial has variance parameter, no need for ScaleKernel
                return KernelFactory.create_kernel('Polynomial', variance=variance, offset=offset, power=power)
                
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")
                
        except Exception as e:
            raise
    
    def generate_lhs_samples(self):
        """Pre-generate LHS samples for all kernel types to ensure good coverage."""
        self.lhs_samples = {}
        # Calculate optimal number of samples per kernel type
        n_kernel_types = 5  # RBF, Matern, Linear, Periodic, Polynomial
        samples_per_kernel = max(20, self.config.max_iter // n_kernel_types)
        
        # RBF kernel samples (angepasste Bounds)
        rbf_bounds = {
            'lengthscale': (0.5, 5.0),
            'variance': (0.5, 2.0)
        }
        self.lhs_samples['RBF'] = build_lhs(rbf_bounds, num_points=samples_per_kernel)
        
        # Matern kernel samples
        matern_bounds = {
            'lengthscale': (0.1, 10.0),
            'nu': (0.5, 5.0),
            'variance': (0.1, 5.0)
        }
        self.lhs_samples['Matern'] = build_lhs(matern_bounds, num_points=samples_per_kernel)
        
        # Linear kernel samples
        linear_bounds = {
            'variance': (0.1, 5.0),
            'offset': (0.0, 2.0)
        }
        self.lhs_samples['Linear'] = build_lhs(linear_bounds, num_points=samples_per_kernel)
        
        # Periodic kernel samples
        periodic_bounds = {
            'lengthscale': (0.1, 10.0),
            'period': (0.1, 10.0),
            'variance': (0.1, 5.0)
        }
        self.lhs_samples['Periodic'] = build_lhs(periodic_bounds, num_points=samples_per_kernel)
        
        # Polynomial kernel samples
        polynomial_bounds = {
            'variance': (0.1, 5.0),
            'offset': (0.0, 2.0),
            'power': (1, 5)
        }
        self.lhs_samples['Polynomial'] = build_lhs(polynomial_bounds, num_points=samples_per_kernel)
        
        # Initialize sample indices
        self.sample_indices = {kernel_type: 0 for kernel_type in self.lhs_samples.keys()}

 