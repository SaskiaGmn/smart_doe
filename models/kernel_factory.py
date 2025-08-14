from gpytorch import kernels
import torch
from typing import Optional

# TODO: what about combined kernels or stacked kernels?

class KernelFactory:

    @staticmethod
    def create_kernel(kernel_type: str, **kwargs) -> kernels.Kernel:
        """
        Create a kernel based on the specified type.
        
        Args:
            kernel_type: Type of kernel ('RBF', 'Matern', 'Linear', 'Periodic', 'Polynomial')
            **kwargs: Kernel-specific parameters
            
        Returns:
            Kernel instance
        """
        # Ensure all parameters are float64, but keep nu as float for Matern
        for key, value in kwargs.items():
            if key == 'nu' and kernel_type == 'Matern':

                kwargs[key] = float(value)
            elif isinstance(value, (int, float)):
                kwargs[key] = torch.tensor(value, dtype=torch.float64)
            elif isinstance(value, torch.Tensor):
                kwargs[key] = value.to(torch.float64)
        
        if kernel_type == 'RBF':
            kernel = kernels.RBFKernel(**kwargs)
        elif kernel_type == 'Matern':
            kernel = kernels.MaternKernel(**kwargs)
        elif kernel_type == 'Linear':
            kernel = kernels.LinearKernel(**kwargs)
        elif kernel_type == 'Periodic':
            kernel = kernels.PeriodicKernel(**kwargs)
        elif kernel_type == 'Polynomial':
            kernel = kernels.PolynomialKernel(**kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        return kernel.to(torch.float64)

    @staticmethod
    def create_task_kernel(kernel_type: str, num_tasks: int, base_kernel: Optional[kernels.Kernel] = None) -> Optional[kernels.Kernel]:
        """
        Create a task kernel for multi-task models.
        
        Args:
            kernel_type: Type of task kernel ('IndexKernel', 'ICM', 'LMC', 'None')
            num_tasks: Number of tasks/outputs
            base_kernel: Base kernel for ICM/LMC kernels
            
        Returns:
            Task kernel
        """
        if kernel_type == 'IndexKernel':
            index_kernel = kernels.IndexKernel(num_tasks=num_tasks)
            return index_kernel.to(torch.float64)
        elif kernel_type == 'ICM':
            # ICM is actually a MultitaskKernel with IndexKernel as task_covar_module
            if base_kernel is None:
                raise ValueError("base_kernel is required for ICM")
            icm_kernel = kernels.MultitaskKernel(
                data_covar_module=base_kernel,
                task_covar_module=kernels.IndexKernel(num_tasks=num_tasks),
                num_tasks=num_tasks,
                rank=1
            )
            return icm_kernel.to(torch.float64)
        elif kernel_type == 'LMC':
            lmc_kernel = kernels.LCMKernel(base_kernels=[base_kernel], num_tasks=num_tasks)
            return lmc_kernel.to(torch.float64)
        elif kernel_type == 'None':
            # No task kernel - return None to indicate separate models
            return None
        else:
            raise ValueError(f"Unknown task kernel type: {kernel_type}")

    @staticmethod
    def create_multi_task_kernel(input_kernel: kernels.Kernel, task_kernel: kernels.Kernel, num_tasks: int):
        """
        Creates a combined kernel for multi-task GP models.
        
        Args:
            input_kernel: The kernel for input dimensions
            task_kernel: The kernel for task dimensions
            num_tasks: Number of tasks
            
        Returns:
            gpytorch.kernels.Kernel: The combined kernel
        """
        # For IndexKernel, create MultitaskKernel
        if isinstance(task_kernel, kernels.IndexKernel):
            multitask_kernel = kernels.MultitaskKernel(
                data_covar_module=input_kernel,
                task_covar_module=task_kernel,
                num_tasks=num_tasks,
                rank=1
            )
            return multitask_kernel.to(torch.float64)
        else:
            return task_kernel.to(torch.float64)