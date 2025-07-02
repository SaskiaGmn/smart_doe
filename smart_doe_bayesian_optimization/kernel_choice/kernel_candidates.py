from typing import List, Dict, Any
from gpytorch.kernels import Kernel
from models.kernel_factory import KernelFactory

class KernelCandidates:
    """
    Defines kernel candidates for automatic kernel selection.
    Focuses on core kernels with different hyperparameter configurations.
    """
    
    @staticmethod
    def get_kernel_candidates() -> List[Dict[str, Any]]:
        """
        Returns a list of kernel configurations to test.
        
        Returns:
            List of dictionaries containing kernel configurations
        """
        return [
            # RBF Kernels with different lengthscales
            {
                'name': 'RBF_small_ls',
                'type': 'RBF',
                'params': {
                    'lengthscale': 0.5,
                    'variance': 1.0
                },
                'description': 'RBF with small lengthscale (0.5)'
            },
            {
                'name': 'RBF_medium_ls',
                'type': 'RBF',
                'params': {
                    'lengthscale': 1.0,
                    'variance': 1.0
                },
                'description': 'RBF with medium lengthscale (1.0)'
            },
            {
                'name': 'RBF_large_ls',
                'type': 'RBF',
                'params': {
                    'lengthscale': 2.0,
                    'variance': 1.0
                },
                'description': 'RBF with large lengthscale (2.0)'
            },
            
            # Matern Kernels with different nu values
            {
                'name': 'Matern_1.5',
                'type': 'Matern',
                'params': {
                    'nu': 1.5,
                    'lengthscale': 1.0,
                    'variance': 1.0
                },
                'description': 'Matern kernel with nu=1.5 (less smooth)'
            },
            {
                'name': 'Matern_2.5',
                'type': 'Matern',
                'params': {
                    'nu': 2.5,
                    'lengthscale': 1.0,
                    'variance': 1.0
                },
                'description': 'Matern kernel with nu=2.5 (more smooth)'
            },
            
            # Linear Kernel
            {
                'name': 'Linear',
                'type': 'Linear',
                'params': {
                    'variance': 1.0,
                    'offset': 0.0
                },
                'description': 'Linear kernel for linear trends'
            },
            
            # Periodic Kernel
            {
                'name': 'Periodic',
                'type': 'Periodic',
                'params': {
                    'lengthscale': 1.0,
                    'period': 2.0,
                    'variance': 1.0
                },
                'description': 'Periodic kernel for periodic patterns'
            },
            
            # Polynomial Kernel
            {
                'name': 'Polynomial',
                'type': 'Polynomial',
                'params': {
                    'power': 2,
                    'variance': 1.0,
                    'offset': 0.0
                },
                'description': 'Polynomial kernel (degree 2)'
            }
        ]
    
    @staticmethod
    def create_kernel_from_config(config: Dict[str, Any]) -> Kernel:
        """
        Creates a kernel instance from a configuration dictionary.
        
        Args:
            config: Dictionary containing kernel configuration
            
        Returns:
            Kernel instance
        """
        kernel_type = config['type']
        params = config['params']
        
        return KernelFactory.create_kernel(kernel_type, **params) 