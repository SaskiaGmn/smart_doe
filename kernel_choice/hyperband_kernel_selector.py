from typing import Dict, List, Any, Tuple, Optional
from gpytorch.kernels import Kernel, ScaleKernel, ProductKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from models.kernel_factory import KernelFactory
from models.gp_model_factory import GPModelFactory, MultitaskGPModel
import torch
import numpy as np
import time
import random
from dataclasses import dataclass
from sampler.lhs import build_lhs
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from sklearn.model_selection import KFold
import gpytorch



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
                 mll_weight: float = 0.7, cv_weight: float = 0.3, num_outputs: int = 1):
        """
        Initialize HyperbandKernelSelector.
        
        Args:
            train_X: Training input data
            train_Y: Training output data
            bounds_list: List of bounds for each input dimension
            scaling_dict: Dictionary with scaling information
            mll_weight: Weight for marginal log-likelihood in scoring, default 0.7
            cv_weight: Weight for cross-validation score in scoring, default 0.3
            num_outputs: Number of outputs (for multi-task models)
        """
        self.train_X = train_X.to(torch.float64)
        self.train_Y = train_Y.to(torch.float64)
        self.bounds_list = bounds_list
        self.scaling_dict = scaling_dict
        self.mll_weight = mll_weight
        self.cv_weight = cv_weight
        self.num_outputs = num_outputs
        self.is_multi_task = num_outputs > 1
        
        if abs(mll_weight + cv_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        #choose likelihood according to the number of outputs
        if self.is_multi_task:
            from gpytorch.likelihoods import MultitaskGaussianLikelihood
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            from gpytorch.likelihoods import GaussianLikelihood
            self.likelihood = GaussianLikelihood()
        
        self.likelihood = self.likelihood.to(torch.float64)
        
        # Use adaptive configuration based on dataset size
        self.config = self.get_adaptive_config()
        #generate lhs samples for the kernel types
        self.sample_indices = {kernel_type: 0 for kernel_type in ['RBF', 'Matern', 'Linear', 'Periodic', 'Polynomial']}
        
        #generate lhs samples for the task kernel types
        if self.is_multi_task:
            task_kernel_types = ['IndexKernel', 'ICM', 'LMC']
            self.task_sample_indices = {task_kernel_type: 0 for task_kernel_type in task_kernel_types}
            
            self.generate_task_lhs_samples()
        else:
            self.generate_lhs_samples()
    
    def get_adaptive_config(self) -> HyperbandConfig:
        """
        Get adaptive configuration based on dataset size and complexity.
        
        Returns:
            HyperbandConfig with appropriate parameters
        """
        n_samples = len(self.train_X)
        n_inputs = self.train_X.shape[1]
        n_outputs = self.num_outputs
        
        print(f"Dataset info: {n_samples} samples, {n_inputs} inputs, {n_outputs} outputs")
        
        if n_samples < 30:
            # Very small datasets: minimal resources, focus on quick evaluation
            print("Using minimal configuration for small dataset")
            return HyperbandConfig(
                max_iter=30,
                eta=2,
                epochs_per_level=[5, 10, 20],
                max_time_seconds=120  # 2 minutes
            )
        elif n_samples < 100:
            # Small datasets: moderate resources
            print("Using moderate configuration for small dataset")
            return HyperbandConfig(
                max_iter=75,
                eta=2,
                epochs_per_level=[10, 20, 40],
                max_time_seconds=300  # 5 minutes
            )
        elif n_samples < 500:
            # Medium datasets: balanced resources
            print("Using balanced configuration for medium dataset")
            return HyperbandConfig(
                max_iter=120,
                eta=3,
                epochs_per_level=[15, 30, 60, 120],
                max_time_seconds=600  # 10 minutes
            )
        else:
            # Large datasets: more resources for thorough search
            print("Using extensive configuration for large dataset")
            return HyperbandConfig(
                max_iter=200,
                eta=3,
                epochs_per_level=[20, 40, 80, 160],
                max_time_seconds=900  # 15 minutes
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
        best_kernel = self.create_kernel_from_config(best_config)
        
        # Store the best task kernel type for later use
        if self.is_multi_task:
            if 'task_kernel_type' not in best_config:
                raise ValueError("Best configuration missing 'task_kernel_type' - optimization may have failed!")
            self.best_task_kernel_type = best_config['task_kernel_type']
            print(f"Best task kernel type selected: {self.best_task_kernel_type}")
        
        return best_kernel
    
    def get_best_task_kernel_type(self) -> str:
        """
        Get the best task kernel type that was selected during optimization.
        
        Returns:
            Best task kernel type (only available for multi-task models)
        """
        if not self.is_multi_task:
            raise ValueError("Task kernel type only available for multi-task models")
        
        if not hasattr(self, 'best_task_kernel_type'):
            raise ValueError("No task kernel type selected yet. Run select_kernel() first.")
        
        return self.best_task_kernel_type
    
    def run_hyperband_optimization(self) -> Dict[str, Any]:
        """
        Run Hyperband optimization to find the best kernel configuration.
        Algorithm based on https://arxiv.org/pdf/1603.06560
        
        Returns:
            Best kernel configuration dictionary
        """
        start_time = time.time()
        best_score = None
        best_config = None
        
        print(f"Starting Hyperband optimization with {self.config.max_iter} max iterations")
        
        # Calculate number of brackets
        s_max = len(self.config.epochs_per_level) - 1
        
        for s in range(s_max, -1, -1):
            # Number of configurations to try in this bracket
            n = int(np.ceil(self.config.max_iter / (s + 1) * (self.config.eta ** s)))
            
            # Number of configurations to keep after each round
            r = self.config.epochs_per_level[0] * (self.config.eta ** s)
            
            print(f"Bracket {s}: {n} configurations, {r} epochs")
            
            # Generate random configurations
            configs = [self.generate_random_kernel_config() for _ in range(n)]
            
            # Successive halving
            for i in range(s + 1):
                # Number of configurations to evaluate in this round
                n_i = int(np.ceil(n / (self.config.eta ** i)))
                # Number of epochs for this round
                r_i = int(r * (self.config.eta ** i))
                
                print(f"  Round {i}: evaluating {n_i} configurations with {r_i} epochs")
                if i == 0:  # Show kernel type distribution in first round
                    kernel_counts = {}
                    for config in configs[:n_i]:
                        if self.is_multi_task:
                            kernel_type = config.get('task_kernel_type', 'Unknown')
                        else:
                            kernel_type = config.get('type', 'Unknown')
                        kernel_counts[kernel_type] = kernel_counts.get(kernel_type, 0) + 1
                    print(f"    Kernel type distribution: {kernel_counts}")
                
                # Evaluate configurations
                scores = []
                for j, config in enumerate(configs[:n_i]):
                    # Check time limit
                    if time.time() - start_time > self.config.max_time_seconds:
                        print(f"Time limit reached ({self.config.max_time_seconds}s)")
                        break
                    
                    score = self.evaluate_kernel_config(config, r_i)
                    scores.append(score)
                    
                    if score != float('-inf') and (best_score is None or score > best_score):
                        best_score = score
                        best_config = config.copy()
                        if self.is_multi_task:
                            print(f"    New best score: {score:.2f} (task kernel: {config.get('task_kernel_type', 'N/A')})")
                        else:
                            print(f"    New best score: {score:.2f} (kernel: {config.get('type', 'N/A')})")
                    elif score == float('-inf'):
                        print(f"    Skipping invalid configuration with -inf score")
                
                # Keep top configurations for next round to reduce computational cost
                if i < s:
                    # Sort by score and keep top 1/eta
                    config_scores = list(zip(configs[:n_i], scores))
                    config_scores.sort(key=lambda x: x[1], reverse=True)
                    configs = [config for config, _ in config_scores[:n_i // self.config.eta]]
        
        print(f"Best score found: {best_score}")
        
        if best_config is None:
            raise ValueError("Hyperband optimization failed to find any valid kernel configuration. This indicates a problem with the data or kernel configurations.")
        
        return best_config
    
    def generate_random_kernel_config(self) -> Dict[str, Any]:
        """
        Generate a kernel configuration using LHS samples.
        For single-task: test different base kernels with hyperparameters
        For multi-task: test different task kernel configurations
        
        Returns:
            Kernel configuration dictionary
        """
        if not self.is_multi_task:
            # Single-task case: test base kernels
            kernel_types = ['RBF', 'Matern', 'Linear', 'Periodic', 'Polynomial']
            #Round robin through the kernel types
            current_total = sum(self.sample_indices.values())
            kernel_type_idx = current_total % len(kernel_types)
            kernel_type = kernel_types[kernel_type_idx]
            
            config = {'type': kernel_type}
            #get the sample for the current kernel type
            sample_idx = self.sample_indices[kernel_type]
            sample = self.lhs_samples[kernel_type][sample_idx]
            
            sample_list = sample.tolist() if hasattr(sample, 'tolist') else list(sample)
            #set the hyperparameters for the current kernel type
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
            
            self.sample_indices[kernel_type] = (self.sample_indices[kernel_type] + 1) % len(self.lhs_samples[kernel_type])
            
            return config
        else:
            # Multi-task case: test task kernel configurations
            task_kernel_types = ['IndexKernel', 'ICM', 'LMC']
            #Round robin through the task kernel types
            current_total = sum(self.task_sample_indices.values())
            task_kernel_idx = current_total % len(task_kernel_types)
            task_kernel_type = task_kernel_types[task_kernel_idx]
            
            config = {'task_kernel_type': task_kernel_type}
            #get the sample for the current task kernel type
            sample_idx = self.task_sample_indices[task_kernel_type]
            sample = self.task_lhs_samples[task_kernel_type][sample_idx]
            
            sample_list = sample.tolist() if hasattr(sample, 'tolist') else list(sample)
            #set the hyperparameters for the current task kernel type
            if task_kernel_type == 'IndexKernel':
                config['variance'] = sample_list[0]
                config['lengthscale'] = sample_list[1]
                config['rank'] = int(sample_list[2])
            elif task_kernel_type == 'ICM':
                config['rank'] = int(sample_list[0])
                config['base_kernel_type'] = ['RBF', 'Matern', 'Linear'][int(sample_list[1]) % 3]
                config['base_kernel_variance'] = sample_list[2]
                if config['base_kernel_type'] == 'RBF':
                    config['base_kernel_lengthscale'] = sample_list[3]
                elif config['base_kernel_type'] == 'Matern':
                    config['base_kernel_lengthscale'] = sample_list[3]
                    config['base_kernel_nu'] = [0.5, 1.5, 2.5][int(sample_list[4]) % 3]
                elif config['base_kernel_type'] == 'Linear':
                    config['base_kernel_offset'] = sample_list[3]
            elif task_kernel_type == 'LMC':
                config['rank'] = int(sample_list[0])
                
                kernel_types = ['RBF', 'Matern', 'Linear']
                base_kernel_type_indices = sample_list[1:1+config['rank']]
                config['base_kernel_types'] = [kernel_types[int(x) % 3] for x in base_kernel_type_indices]
                
                config['base_kernel_variances'] = sample_list[1+config['rank']:1+2*config['rank']]
                config['base_kernel_lengthscales'] = sample_list[1+2*config['rank']:1+3*config['rank']]
                
                for i in range(config['rank']):
                    config[f'base_kernel_variance_{i+1}'] = config['base_kernel_variances'][i]
                    config[f'base_kernel_lengthscale_{i+1}'] = config['base_kernel_lengthscales'][i]
                
                config['base_kernel_nus'] = []
                nu_start_index = 1 + 3 * config['rank']
                for i, kernel_type in enumerate(config['base_kernel_types']):
                    if kernel_type == 'Matern':
                        raw_value = sample_list[nu_start_index + i] if nu_start_index + i < len(sample_list) else 0
                        nu = [0.5, 1.5, 2.5][int(raw_value) % 3]
                        config['base_kernel_nus'].append(nu)
            
            self.task_sample_indices[task_kernel_type] = (self.task_sample_indices[task_kernel_type] + 1) % len(self.task_lhs_samples[task_kernel_type])
            
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
            if not self.is_multi_task:
                kernel = self.create_kernel_from_config(config)
                return self.evaluate_single_task_model(kernel, epochs, config)
            else:
                kernel = self.create_kernel_from_config(config)
                return self.evaluate_multitask_model(kernel, epochs, config)
                
        except Exception as e:
            print(f"      Error in evaluate_kernel_config: {str(e)}")
            return float('-inf')
    
    def evaluate_single_task_model(self, kernel, epochs: int, config: Dict[str, Any]) -> float:
        """Evaluate a single SingleTaskGP model."""
        from gpytorch.likelihoods import GaussianLikelihood
        likelihood = GaussianLikelihood()
        
        outcome_transform = Standardize(m=self.train_Y.shape[1])
        input_transform = Normalize(d=self.train_X.shape[1])
        
        gp_model = GPModelFactory.create_model(
            model_type="SingleTaskGP",
            train_X=self.train_X,
            train_Y=self.train_Y,
            kernel=kernel,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform
        )
        #Marginal Log-Likelihood for the model 
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        trained_epochs = self.train_main_model_with_early_stopping(gp_model, mll, epochs)
        
        with torch.no_grad():
            output = gp_model(gp_model.train_inputs[0])
            #Compute the Marginal Log-Likelihood for the model to get an estimation of the model performance (Fit of model to the training data)
            mll_score = -mll(output, gp_model.train_targets).item()
            
            if torch.isnan(torch.tensor(mll_score)) or torch.isinf(torch.tensor(mll_score)):
                raise RuntimeError(f"Invalid MLL score detected: {mll_score}. This indicates a numerical problem in the Single-Task GP model.")
            
            # CV temporarily disabled due to gradient issues
            #CV should be computed with the same kernel as the one used for the MLL score to get a fair comparison of the model performance
            # TODO: Fix CV implementation
            combined_score = mll_score
            
            return combined_score
    
    def get_adaptive_training_config(self, n_samples: int) -> dict:
        """
        Get adaptive training configuration based on dataset size.
        """
        if n_samples < 10:
            return {
                'max_epochs': 30, 'patience': 5, 'lr': 0.05, 'min_epochs': 10
            }
        elif n_samples < 20:
            return {
                'max_epochs': 50, 'patience': 8, 'lr': 0.1, 'min_epochs': 15
            }
        elif n_samples < 50:
            return {
                'max_epochs': 80, 'patience': 10, 'lr': 0.1, 'min_epochs': 20
            }
        else:
            return {
                'max_epochs': 120, 'patience': 15, 'lr': 0.1, 'min_epochs': 25
            }
    
    def train_main_model_with_early_stopping(self, gp_model, mll: ExactMarginalLogLikelihood, max_epochs: int) -> int:
        """
        Train main GP model with early stopping based on convergence detection. Early stopping is used to avoid overfitting.
        
        Args:
            gp_model: GP model to train (SingleTaskGP or MultiTaskGP object)
            mll: Marginal Log-Likelihood for the model
            max_epochs: Maximum number of epochs
            
        Returns:
            Number of epochs actually trained
        """
        # Get adaptive training configuration based on dataset size
        n_samples = self.train_X.shape[0]
        adaptive_config = self.get_adaptive_training_config(n_samples)
        
        # Use adaptive parameters, but respect max_epochs as upper bound
        max_epochs = min(max_epochs, adaptive_config['max_epochs'])
        patience = adaptive_config['patience']
        lr = adaptive_config['lr']
        min_epochs = adaptive_config['min_epochs']
        
        from models.optimizer_factory import OptimizerFactory
        optimizer = OptimizerFactory.create_optimizer(
            type="adam",
            model_parameters=gp_model.parameters(),
            lr=lr
        )
        
        gp_model.train()
        
        # Early stopping parameters
        min_improvement = 1e-4 
        best_loss = float('inf')
        patience_counter = 0
        
        # Store loss history for convergence detection
        loss_history = []
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            output = gp_model(gp_model.train_inputs[0])
            loss = -mll(output, gp_model.train_targets)
            if loss.dim() > 0:
                loss = loss.sum()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            if current_loss < best_loss - min_improvement:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping conditions
            if patience_counter >= patience:
                # Check if we have enough epochs for meaningful training
                if epoch >= min_epochs:
                    return epoch + 1
            
            # Additional convergence check: loss stability
            if len(loss_history) >= 15:
                recent_losses = loss_history[-15:]
                loss_std = np.std(recent_losses)
                if loss_std < 1e-5:  
                    return epoch + 1
        
        return max_epochs
    
    def get_adaptive_cv_config(self, n_samples: int) -> dict:
        """
        Get adaptive CV configuration based on dataset size. Can be used for both single-task and multi-task models. Configuration is just a suggestion, the user can change the parameters if needed.
        
        Args:
            n_samples: Number of samples in the dataset
            
        Returns:
            Dictionary with CV configuration parameters
        """
        if n_samples < 10:
            return {
                'k_folds': 2,
                'max_epochs': 15,
                'patience': 3,
                'lr': 0.05,
                'min_fold_size': 3
            }
        elif n_samples < 20:
            return {
                'k_folds': 3,
                'max_epochs': 20,
                'patience': 4,
                'lr': 0.08,
                'min_fold_size': 4
            }
        elif n_samples < 50:
            return {
                'k_folds': 4,
                'max_epochs': 25,
                'patience': 5,
                'lr': 0.1,
                'min_fold_size': 5
            }
        else:
            return {
                'k_folds': 5,
                'max_epochs': 30,
                'patience': 7,
                'lr': 0.1,
                'min_fold_size': 8
            }
    

    

    
    def compute_kfold_cv_score(self, kernel_config):
        """
        Compute K-Fold Cross-Validation score using GPyTorch Batch Mode.
        
        Args:
            kernel_config: Kernel configuration dictionary
            
        Returns:
            Average K-Fold CV score (NLPD)
        """
        try:
            from sklearn.model_selection import KFold
            import numpy as np
            
            # Convert torch tensors to numpy for splitting
            X = self.train_X.cpu().numpy()
            Y = self.train_Y.cpu().numpy()
            
            n = X.shape[0]
            
            # Get adaptive CV configuration
            cv_config = self.get_adaptive_cv_config(n)
            k_folds = cv_config['k_folds']
            min_fold_size = cv_config['min_fold_size']
            
            print(f"      Using GPyTorch Batch CV: {k_folds} folds")
            
            # Ensure minimum fold size
            if n // k_folds < min_fold_size:
                k_folds = max(2, n // min_fold_size)
                print(f"      Adjusted to {k_folds} folds for minimum fold size {min_fold_size}")
            
            # Create kernel once (will be shared across folds)
            kernel = self.create_kernel_from_config(kernel_config)
            
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            nlpd_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                try:
                    print(f"      CV Fold {fold_idx + 1}: Training Batch GP...")
                    
                    # Create tensors with proper gradient tracking
                    train_X = torch.tensor(X[train_idx], dtype=torch.float64, requires_grad=False)
                    train_Y = torch.tensor(Y[train_idx], dtype=torch.float64, requires_grad=False)
                    val_X = torch.tensor(X[val_idx], dtype=torch.float64, requires_grad=False)
                    val_Y = torch.tensor(Y[val_idx], dtype=torch.float64, requires_grad=False)
                    
                    # Create batch-mode model
                    if not self.is_multi_task:
                        from gpytorch.likelihoods import GaussianLikelihood
                        likelihood = GaussianLikelihood().to(torch.float64)
                        
                        # Create transforms for CV
                        outcome_transform = Standardize(m=train_Y.shape[1])
                        input_transform = Normalize(d=train_X.shape[1])
                        
                        # Initialize transforms with training data
                        outcome_transform.train()
                        _ = outcome_transform(train_Y)
                        outcome_transform.eval()
                        
                        input_transform.train()
                        _ = input_transform(train_X)
                        input_transform.eval()
                        
                        # Create batch-mode GP model
                        from models.gp_model_factory import GPModelFactory
                        gp_model = GPModelFactory.create_model(
                            model_type="SingleTaskGP",
                            train_X=train_X,
                            train_Y=train_Y,
                            kernel=kernel,
                            likelihood=likelihood,
                            outcome_transform=outcome_transform,
                            input_transform=input_transform
                        )
                    else:
                        from gpytorch.likelihoods import MultitaskGaussianLikelihood
                        likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_outputs).to(torch.float64)
                        
                        # Create transforms for CV
                        outcome_transform = Standardize(m=train_Y.shape[1])
                        input_transform = Normalize(d=train_X.shape[1])
                        
                        # Initialize transforms with training data
                        outcome_transform.train()
                        _ = outcome_transform(train_Y)
                        outcome_transform.eval()
                        
                        input_transform.train()
                        _ = input_transform(train_X)
                        input_transform.eval()
                        
                        # Create batch-mode MultiTask GP model
                        from models.gp_model_factory import GPModelFactory
                        gp_model = GPModelFactory.create_model(
                            model_type="MultiTaskGP",
                            train_X=train_X,
                            train_Y=train_Y,
                            kernel=kernel,
                            likelihood=likelihood,
                            outcome_transform=outcome_transform,
                            input_transform=input_transform
                        )
                    
                    # Set up MLL and optimizer
                    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
                    optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
                    
                    # Training loop
                    gp_model.train()
                    likelihood.train()
                    
                    for epoch in range(cv_config['max_epochs']):
                        optimizer.zero_grad()
                        output = gp_model(train_X)
                        loss = -mll(output, train_Y)
                        
                        # Ensure loss is scalar for backward pass
                        if loss.dim() > 0:
                            loss = loss.sum()
                        
                        loss.backward()
                        optimizer.step()
                    
                    # Evaluation
                    gp_model.eval()
                    likelihood.eval()
                    
                    with torch.no_grad():
                        output = gp_model(val_X)
                        mu = output.mean
                        var = output.variance
                        
                        nlpd_score = self.calculate_nlpd(val_Y, mu, var)
                        nlpd_scores.append(nlpd_score)
                    
                    print(f"      CV Fold {fold_idx + 1}: NLPD = {nlpd_score:.4f}")
                    
                except Exception as e:
                    print(f"      CV Fold {fold_idx + 1} failed: {str(e)}")
                    nlpd_scores.append(float('inf'))  # Higher is worse for NLPD
            
            if len(nlpd_scores) == 0:
                print(f"      All CV folds failed. Returning inf")
                return float('inf')
            
            avg_nlpd = np.mean(nlpd_scores)
            print(f"      Average CV NLPD: {avg_nlpd:.4f}")
            
            return avg_nlpd
            
        except Exception as e:
            print(f"      CV computation failed: {str(e)}")
            return float('inf')
    

    
    def calculate_nlpd(self, y_true, y_pred, y_var):
        """Calculate Negative Log Predictive Density."""
        import torch
        
        eps = 1e-8
        y_var = torch.clamp(y_var, min=eps)
        
        log_prob = -0.5 * torch.log(2 * torch.pi * y_var) - 0.5 * (y_true - y_pred)**2 / y_var
        
        return torch.mean(-log_prob).item()
    
    def create_kernel_from_config(self, config: Dict[str, Any]) -> Kernel:
        """
        Create a kernel from configuration dictionary.
        
        Args:
            config: Kernel configuration dictionary
            
        Returns:
            Kernel instance (input kernel for single-task, multitask kernel for multi-task)
        """
        if not self.is_multi_task:
            # Single-task case: create base kernel
            kernel_type = config['type']
            
            if kernel_type == 'RBF':
                lengthscale = float(config['lengthscale'])
                variance = float(config['variance'])
                
                kernel = KernelFactory.create_kernel('RBF', lengthscale=lengthscale)
                return ScaleKernel(kernel, outputscale=variance)
                
            elif kernel_type == 'Matern':
                lengthscale = float(config['lengthscale'])
                nu = float(config['nu'])
                variance = float(config['variance'])
                
                kernel = KernelFactory.create_kernel('Matern', lengthscale=lengthscale, nu=nu)
                return ScaleKernel(kernel, outputscale=variance)
                
            elif kernel_type == 'Linear':
                variance = float(config['variance'])
                offset = float(config['offset'])
                
                return KernelFactory.create_kernel('Linear', variance=variance, offset=offset)
                
            elif kernel_type == 'Periodic':
                lengthscale = float(config['lengthscale'])
                period = float(config['period'])
                variance = float(config['variance'])
                
                kernel = KernelFactory.create_kernel('Periodic', lengthscale=lengthscale, period=period)
                return ScaleKernel(kernel, outputscale=variance)
                
            elif kernel_type == 'Polynomial':
                variance = float(config['variance'])
                offset = float(config['offset'])
                power = int(config['power'])  
                
                return KernelFactory.create_kernel('Polynomial', variance=variance, offset=offset, power=power)
                
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")
        else:
            # Multi-task case: create multitask kernel
            task_kernel_type = config['task_kernel_type']
            
            if task_kernel_type == 'IndexKernel':
                # IndexKernel: create a proper multitask kernel using MultitaskKernel
                from gpytorch.kernels import RBFKernel, MultitaskKernel
                
                # Create input kernel (use RBF as default)
                input_kernel = RBFKernel().to(torch.float64)
                
                task_kernel = MultitaskKernel(
                    input_kernel,
                    num_tasks=self.num_outputs
                ).to(torch.float64)
                
                return task_kernel
                
            elif task_kernel_type == 'ICM':
                from gpytorch.kernels import MultitaskKernel
                
                base_kernel_type = config['base_kernel_type']
                if base_kernel_type == 'RBF':
                    from gpytorch.kernels import RBFKernel
                    base_kernel = RBFKernel(
                        lengthscale=config['base_kernel_lengthscale'],
                        variance=config['base_kernel_variance']
                    ).to(torch.float64)
                elif base_kernel_type == 'Matern':
                    from gpytorch.kernels import MaternKernel
                    base_kernel = MaternKernel(
                        lengthscale=config['base_kernel_lengthscale'],
                        nu=config['base_kernel_nu'],
                        variance=config['base_kernel_variance']
                    ).to(torch.float64)
                elif base_kernel_type == 'Linear':
                    from gpytorch.kernels import LinearKernel
                    base_kernel = LinearKernel(
                        variance=config['base_kernel_variance'],
                        offset=config['base_kernel_offset']
                    ).to(torch.float64)
                else:
                    raise ValueError(f"Unknown base kernel type: {base_kernel_type}")
                
                task_kernel = MultitaskKernel(
                    base_kernel,
                    num_tasks=self.num_outputs
                ).to(torch.float64)
                
                return task_kernel
                
            elif task_kernel_type == 'LMC':
                from gpytorch.kernels import MultitaskKernel, RBFKernel
                
                base_kernel = RBFKernel(
                    lengthscale=config['base_kernel_lengthscale_1'],
                    variance=config['base_kernel_variance_1']
                ).to(torch.float64)
                
                task_kernel = MultitaskKernel(
                    base_kernel,
                    num_tasks=self.num_outputs
                ).to(torch.float64)
                
                return task_kernel
                
            else:
                raise ValueError(f"Unknown task kernel type: {task_kernel_type}")
    
    def generate_lhs_samples(self):
        """Generate LHS samples for different kernel types. Can be adapted to the user's needs."""
        kernel_params = {
            'RBF': {
                'lengthscale': (0.1, 10.0),
                'variance': (0.1, 10.0)
            },
            'Linear': {
                'variance': (0.1, 10.0),
                'offset': (0.0, 5.0)
            },
            'Periodic': {
                'lengthscale': (0.1, 10.0),
                'period': (0.1, 10.0),
                'variance': (0.1, 10.0)
            },
            'Polynomial': {
                'variance': (0.1, 10.0),
                'offset': (0.0, 5.0),
                'power': (1, 5)
            }
        }
        
        self.lhs_samples = {}
        
        for kernel_type, params in kernel_params.items():
            try:
                bounds = {f'{param}_range': param_range for param, param_range in params.items()}
                samples = build_lhs(bounds, self.config.max_iter)
                self.lhs_samples[kernel_type] = samples
                print(f"Generated {len(samples)} LHS samples for {kernel_type}")
            except Exception as e:
                print(f"Error generating LHS samples for {kernel_type}: {e}")
                raise RuntimeError(f"Failed to generate LHS samples for {kernel_type}. LHS sampling is required for proper kernel optimization.")
        
        # Special handling for Matern: generate discrete nu values as they can only take 0.5, 1.5, 2.5
        # Generate LHS samples for lengthscale and variance
        matern_bounds = {
            'lengthscale_range': (0.1, 10.0),
            'variance_range': (0.1, 10.0)
        }
        matern_samples = build_lhs(matern_bounds, self.config.max_iter)
        
        # Add discrete nu values (0.5, 1.5, 2.5) by round-robin method 
        valid_nu_values = [0.5, 1.5, 2.5]
        nu_values = []
        for i in range(self.config.max_iter):
            nu_values.append(valid_nu_values[i % 3])
        
        # Combine samples: [lengthscale, nu, variance]
        matern_combined = torch.zeros(self.config.max_iter, 3, dtype=torch.float64)
        matern_combined[:, 0] = matern_samples[:, 0]
        matern_combined[:, 1] = torch.tensor(nu_values, dtype=torch.float64)
        matern_combined[:, 2] = matern_samples[:, 1]
        
        self.lhs_samples['Matern'] = matern_combined
        print(f"Generated {len(matern_combined)} Matern samples with discrete nu values")

    def generate_task_lhs_samples(self):
        """Generate LHS samples for different task kernel types. Can be adapted to the user's needs."""
        task_kernel_params = {
            'IndexKernel': {
                'variance': (0.1, 10.0),
                'lengthscale': (0.1, 10.0),
                'rank': (1, 3) 
            },
            'ICM': {
                'rank': (1, 3),
                'base_kernel_type': (0, 3),
                'base_kernel_variance': (0.1, 10.0),
                'base_kernel_lengthscale': (0.1, 10.0),
                'base_kernel_nu': (0, 3),
                'base_kernel_offset': (0.0, 5.0)
            },
            'LMC': {
                'rank': (1, 3),
                'base_kernel_type_1': (0, 3), 
                'base_kernel_type_2': (0, 3),
                'base_kernel_type_3': (0, 3),
                'base_kernel_variance_1': (0.1, 10.0),
                'base_kernel_variance_2': (0.1, 10.0),
                'base_kernel_variance_3': (0.1, 10.0),
                'base_kernel_lengthscale_1': (0.1, 10.0),
                'base_kernel_lengthscale_2': (0.1, 10.0),
                'base_kernel_lengthscale_3': (0.1, 10.0),
                'base_kernel_nu_1': (0, 3),
                'base_kernel_nu_2': (0, 3),
                'base_kernel_nu_3': (0, 3)
            }
        }
        
        self.task_lhs_samples = {}
        
        for task_kernel_type, params in task_kernel_params.items():
            try:
                bounds = {f'{param}_range': param_range for param, param_range in params.items()}
                samples = build_lhs(bounds, self.config.max_iter)
                self.task_lhs_samples[task_kernel_type] = samples
                print(f"Generated {len(samples)} LHS samples for task kernel {task_kernel_type}")
            except Exception as e:
                print(f"Error generating LHS samples for task kernel {task_kernel_type}: {e}")
                raise RuntimeError(f"Failed to generate LHS samples for task kernel {task_kernel_type}. LHS sampling is required for proper kernel optimization.")

    def evaluate_multitask_model(self, kernel, epochs: int, config: Dict[str, Any]) -> float:
        """Evaluate a GPyTorch MultitaskGP model with explicit kernel."""
        from gpytorch.likelihoods import MultitaskGaussianLikelihood
        likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_outputs)
        
        # Create transforms
        from botorch.models.transforms.outcome import Standardize
        from botorch.models.transforms.input import Normalize
        outcome_transform = Standardize(m=self.train_Y.shape[1])
        input_transform = Normalize(d=self.train_X.shape[1])
        
        from models.gp_model_factory import GPModelFactory
        gp_model = GPModelFactory.create_model(
            model_type="MultiTaskGP",
            train_X=self.train_X,
            train_Y=self.train_Y,
            kernel=kernel,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=input_transform
        )
        
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        trained_epochs = self.train_main_model_with_early_stopping(gp_model, mll, epochs)
        
        with torch.no_grad():
            output = gp_model(gp_model.train_inputs[0])
            mll_score = -mll(output, gp_model.train_targets).item()
            
            if torch.isnan(torch.tensor(mll_score)) or torch.isinf(torch.tensor(mll_score)):
                raise RuntimeError(f"Invalid MLL score detected: {mll_score}. This indicates a numerical problem in the Multi-Task GP model.")
            
            # CV temporarily disabled due to gradient issues
            # TODO: Fix CV implementation
            combined_score = mll_score
            
            return combined_score 

 