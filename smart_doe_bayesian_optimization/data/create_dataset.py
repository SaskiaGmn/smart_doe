import torch
from typing import Callable, Dict, Tuple
from sampler import build_lhs, build_space_filling_lhs, build_fractional_factorial, build_full_factorial, build_taguchi

# TODO: implement transform data function
# TODO: implement receive dataset from filepath function
# TODO: Bounds integration, especially for the filepath acquiring of data
# TODO: further testing, especially for multi input/output functions
# TODO: what about more robust scaling methods? What about the scaling schedule?

# TODO: the **kwargs should be changed here into a range dict, which is a more clear way of initiating it

class DatasetManager:
    def __init__(self, dtype: torch.dtype, filepath: str = None, num_input_dimensions: int = 1) -> None:
        self.dtype = dtype
        self.filepath = filepath
        self.num_input_dimensions = num_input_dimensions
        self.input_dim = num_input_dimensions
        self.output_dim = 1  # May be changed to a list of dimensions
        self.unscaled_data = None
        self.bounds_list = []

    def func_create_dataset(self, dataset_func: Callable[..., torch.Tensor], num_datapoints: int, 
                          sampling_method: str = "grid", noise_level: float = 0.0, **kwargs):
        """
        Erstellt einen Datensatz basierend auf einer Funktion.
        
        Args:
            dataset_func: Funktion, die die Ausgabewerte berechnet
            num_datapoints: Anzahl der zu erstellenden Datenpunkte
            sampling_method: Methode zur Stichprobenziehung ("grid", "random", "lhs", "space_filling_lhs", "fractional_factorial", "full_factorial")
            noise_level: Standardabweichung des Rauschens
            **kwargs: Zusätzliche Parameter für die Stichprobenziehung
        """
        self.num_datapoints = num_datapoints

        # Create bounds dictionary from kwargs
        bounds = {}
        for key, value in kwargs.items():
            if "range" in key:
                bounds[key] = value
        
        # Choose the corresponding sampling method
        if sampling_method == "random":
            inputs = self._random_sampling(bounds, num_datapoints)
        elif sampling_method == "grid":
            inputs = self._grid_sampling(bounds, num_datapoints)
        elif sampling_method == "lhs":
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
            raise ValueError("Sampling method must be one of: 'random', 'grid', 'lhs', 'space_filling_lhs', 'fractional_factorial', 'full_factorial', 'taguchi'")

        self.setbounds(**kwargs)

        inputs = inputs.to(self.dtype)
        outputs, self.output_dim = dataset_func(inputs)

        if noise_level > 0:
            outputs = self.add_noise(outputs=outputs, noise_level=noise_level)

        outputs = outputs.to(self.dtype)

        self.unscaled_data = (inputs.clone(), outputs.clone())
        self.check_shape(inputs, outputs)
        self.check_dimensions(inputs, outputs)

    def _random_sampling(self, bounds: Dict[str, Tuple[float, float]], num_points: int) -> torch.Tensor:
        """Creates random samples."""
        inputs = []
        for range_val in bounds.values():
            inputs.append(torch.rand(num_points) * (range_val[1] - range_val[0]) + range_val[0])
        return torch.stack(inputs, dim=1)

    def _grid_sampling(self, bounds: Dict[str, Tuple[float, float]], num_points: int) -> torch.Tensor:
        """Creates samples on a regular grid."""
        inputs = []
        for range_val in bounds.values():
            inputs.append(torch.linspace(range_val[0], range_val[1], steps=num_points))
        return torch.stack(inputs, dim=1)

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