import numpy as np
import torch
from typing import Dict, Tuple, List
from pyDOE3 import ff2n

def build_full_factorial(bounds: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    """
    Creates a full factorial design for the given parameter bounds.
    Uses pyDOE3 for the generation of the design.
    
    Args:
        bounds: Dictionary with parameter names as keys and (min, max) tuples as values
        
    Returns:
        torch.Tensor: Tensor with the generated points
    """
    num_dimensions = len(bounds)
    
    samples = ff2n(num_dimensions)
    
    # Scale the values to the actual bounds
    # Formula: scaled = ((x + 1) / 2) * (max - min) + min: Transforms [-1, 1] -> [min, max] for each parameter
    scaled_samples = np.zeros_like(samples, dtype=float)
    for i, (param_name, (min_val, max_val)) in enumerate(bounds.items()):
        scaled_samples[:, i] = (samples[:, i] + 1) / 2 * (max_val - min_val) + min_val
    
    return torch.tensor(scaled_samples, dtype=torch.float64) 