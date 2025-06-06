import numpy as np
import torch
from typing import Dict, Tuple, List
from pyDOE import lhs

def build_space_filling_lhs(bounds: Dict[str, Tuple[float, float]], num_points: int) -> torch.Tensor:
    """
    Creates space-filling Latin Hypercube Samples for the given parameter bounds.
    
    Args:
        bounds: Dictionary with parameter names as keys and (min, max) tuples as values
        num_points: Number of points to generate
        
    Returns:
        torch.Tensor: Tensor with the generated points, Shape (num_points, num_dimensions)
    """
    num_dimensions = len(bounds)
    
    # Create LHS with pyDOE and maximin criterion
    samples = lhs(num_dimensions, samples=num_points, criterion='maximin')
    
    # Scale the samples to the actual bounds
    for i, (param_name, (min_val, max_val)) in enumerate(bounds.items()):
        samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
    
    return torch.tensor(samples, dtype=torch.float64) 