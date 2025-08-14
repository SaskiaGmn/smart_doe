import torch
from typing import Callable, Dict, Tuple
from sampler import build_lhs, build_space_filling_lhs, build_fractional_factorial, build_full_factorial, build_taguchi

# TODO: implement transform data function
# TODO: implement receive dataset from filepath function
# TODO: what about more robust scaling methods?

class DatasetManager:
    def __init__(self, dtype: torch.dtype, filepath: str = None, num_input_dimensions: int = 1, num_output_dimensions: int = 1) -> None:
        self.dtype = dtype
        self.filepath = filepath
        self.num_input_dimensions = num_input_dimensions
        self.num_output_dimensions = num_output_dimensions
        self.input_dim = num_input_dimensions
        self.output_dim = num_output_dimensions  
        self.unscaled_data = None
        self.bounds_list = []

    def func_create_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints: int, 
                          sampling_method: str = "grid", noise_level: float = 0.0, **kwargs):
        """
        Creates a dataset based on a function.
        
        Args:
            dataset_func: Function that computes the output values
            num_datapoints: Number of data points to create
            sampling_method: Sampling method ("grid", "random", "lhs", "space_filling_lhs", "fractional_factorial", "full_factorial")
            noise_level: Standard deviation of the noise
            **kwargs: Additional parameters for the sampling method
        """
        self.num_datapoints = num_datapoints

        # Create bounds dictionary from kwargs
        bounds = {}
        for key, value in kwargs.items():
            if "range" in key:
                bounds[key] = value
        
        # Choose the corresponding sampling method
        if sampling_method == "lhs":
            inputs = build_lhs(bounds, num_datapoints)
        elif sampling_method == "space_filling_lhs":
            inputs = build_space_filling_lhs(bounds, num_datapoints)
        elif sampling_method == "fractional_factorial":
            inputs = build_fractional_factorial(bounds, main_factors=kwargs.get('main_factors', 3))
        elif sampling_method == "full_factorial":
            inputs = build_full_factorial(bounds)
        elif sampling_method == "taguchi":
            inputs = build_taguchi(bounds)
        else:
            raise ValueError("Sampling method must be one of: 'lhs', 'space_filling_lhs', 'fractional_factorial', 'full_factorial', 'taguchi'")

        self.setbounds(**kwargs)

        inputs = inputs.to(self.dtype)
        
        # Handle multi-output functions
        if hasattr(dataset_func, '__name__') and 'multi_output' in dataset_func.__name__:
            # For multi-output functions, pass the number of outputs
            outputs, self.output_dim = dataset_func(inputs, num_outputs=self.num_output_dimensions)
        else:
            # For single-output functions, use the original behavior
            outputs, self.output_dim = dataset_func(inputs)

        if noise_level > 0:
            outputs = self.add_noise(outputs=outputs, noise_level=noise_level)

        outputs = outputs.to(self.dtype)

        self.unscaled_data = (inputs.clone(), outputs.clone())
        self.check_shape(inputs, outputs)
        self.check_dimensions(inputs, outputs)



    def add_noise(self, outputs: torch.Tensor, noise_level: float):
        """ 
        Adds Gaussian noise to the output data.

        Args:
            outputs: Ausgabedaten
            noise_level: Standardabweichung des Rauschens
        """
        return outputs + noise_level * torch.randn_like(outputs)

    def setbounds(self, **kwargs):
        """Sets the bounds based on the passed ranges."""
        self.bounds_list = [value for key, value in kwargs.items() if "range" in key]

    def check_shape(self, inputs: torch.Tensor, outputs: torch.Tensor):
        if inputs.shape[1] != self.num_input_dimensions:
            raise ValueError(f"Input data has {inputs.shape[1]} dimensions, but {self.num_input_dimensions} expected")
        if outputs.shape[1] != self.output_dim:
            raise ValueError(f"Output data has {outputs.shape[1]} dimensions, but {self.output_dim} expected")

    def check_dimensions(self, inputs: torch.Tensor, outputs: torch.Tensor):
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError("Number of input and output points does not match")

    def train_test_split(self):
        pass