from gpytorch import kernels
import torch

# TODO: what about combined kernels or stacked kernels?

class KernelFactory:

    @staticmethod
    def create_kernel(kernel_type: str, **kwargs):
        """
        Creates a kernel based on the specified type and parameters.
        
        Args:
            kernel_type (str): The type of kernel to create. Examples include 'RBF', 'Matern', 'Periodic', etc.
            **kwargs: Arbitrary keyword arguments, mainly for kernel configuration such as lengthscale.
        
        Returns:
            gpytorch.kernels.Kernel: The instantiated kernel object.
        
        Raises:
            ValueError: If the kernel type is unknown or parameters are missing.
        """
        if kernel_type == 'RBF':
            # Map our parameter names to gpytorch parameter names
            lengthscale = kwargs.get('lengthscale', 1.0)
            return kernels.RBFKernel(lengthscale=lengthscale)
        elif kernel_type == 'Matern':
            # Map our parameter names to gpytorch parameter names
            nu = kwargs.get('nu', 2.5)
            lengthscale = kwargs.get('lengthscale', 1.0)
            return kernels.MaternKernel(nu=nu, lengthscale=lengthscale)
        elif kernel_type == 'Periodic':
            # Map our parameter names to gpytorch parameter names
            lengthscale = kwargs.get('lengthscale', 1.0)
            period = kwargs.get('period', 1.0)
            return kernels.PeriodicKernel(lengthscale=lengthscale, period=period)
        elif kernel_type == 'Linear':
            # Map our parameter names to gpytorch parameter names
            variance = kwargs.get('variance', 1.0)
            offset = kwargs.get('offset', 0.0)
            return kernels.LinearKernel(variance=variance, offset=offset)
        elif kernel_type == 'Polynomial':
            # Map our parameter names to gpytorch parameter names
            power = kwargs.get('power', 2)
            variance = kwargs.get('variance', 1.0)
            offset = kwargs.get('offset', 0.0)
            return kernels.PolynomialKernel(power=power, variance=variance, offset=offset)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")