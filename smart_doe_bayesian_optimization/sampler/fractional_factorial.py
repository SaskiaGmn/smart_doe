import numpy as np
import torch
from typing import Dict, Tuple, List
from pyDOE3 import ff2n, fracfact
import itertools

def build_fractional_factorial(bounds: Dict[str, Tuple[float, float]], main_factors: int = 3) -> torch.Tensor:
    """
    Erstellt ein Fractional Factorial Design für die gegebenen Parameter-Grenzen.
    Verwendet pyDOE3 für die Generierung des Designs.
    
    Args:
        bounds: Dictionary mit Parameternamen als Schlüssel und (min, max) Tupeln als Werte
        
    Returns:
        torch.Tensor: Tensor mit den generierten Punkten
    """

    num_dimensions = len(bounds)
    max_possible_interactions = sum(
        len(list(itertools.combinations(range(main_factors), r)))
        for r in range(2, main_factors + 1)
    )

    num_generated = num_dimensions - main_factors
    if max_possible_interactions < num_generated:
        raise ValueError("Design is not possible with the given number of main factors")

    # Create aliases A, B, C, D, ...
    factors = [chr(65 + i) for i in range(num_dimensions)]  # ['A', 'B', 'C', 'D']
    if main_factors >= num_dimensions:
        raise ValueError("Number of main factors must be less than total factors")

    # Name of main factors: A, B, C, ...
    letters = [chr(65 + i) for i in range(main_factors)]  # 'A', 'B', ...

    # Initial generator: main factors as own columns
    generator = list(letters)

    # Number of columns to generate
    num_generated = num_dimensions - main_factors

    # Generate interactions from main factors (length 2)
    interactions = []
    for r in reversed(range(2, main_factors + 1)):  # highest order interactions first
        for combo in itertools.combinations(letters, r):
            interactions.append(''.join(combo))
            if len(interactions) == num_generated:
                break
        if len(interactions) == num_generated:
            break

    # Final generator string
    full_generator = generator + interactions
    gen = ' '.join(full_generator)
            
    samples = fracfact(gen)
    
    # Scale the values to the actual bounds
    # Formula: scaled = ((x + 1) / 2) * (max - min) + min: Transforms [-1, 1] -> [min, max] for each parameter
    scaled_samples = np.zeros_like(samples, dtype=float)
    for i, (param_name, (min_val, max_val)) in enumerate(bounds.items()):
        scaled_samples[:, i] = (samples[:, i] + 1) / 2 * (max_val - min_val) + min_val
    
    return torch.tensor(scaled_samples, dtype=torch.float64) 