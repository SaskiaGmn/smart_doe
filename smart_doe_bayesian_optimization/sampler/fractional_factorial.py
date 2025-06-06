import numpy as np
import torch
from typing import Dict, Tuple, List
from pyDOE import ff2n, fracfact

def build_fractional_factorial(bounds: Dict[str, Tuple[float, float]], resolution: int = 3) -> torch.Tensor:
    """
    Erstellt ein Fractional Factorial Design für die gegebenen Parameter-Grenzen.
    Verwendet pyDOE für die Generierung des Designs.
    
    Args:
        bounds: Dictionary mit Parameternamen als Schlüssel und (min, max) Tupeln als Werte
        resolution: Auflösung des Designs (3 = Haupteffekte, 4 = Haupteffekte + Zweifachwechselwirkungen)
        
    Returns:
        torch.Tensor: Tensor mit den generierten Punkten
    """
    num_dimensions = len(bounds)
    
    if num_dimensions <= 2:
        # For 2 or less factors we use a full 2^k design
        samples = ff2n(num_dimensions)
    else:
        # For more factors we use a fractional factorial design
        # Create the Alias notation for the factors (A, B, C, ...)
        factors = [chr(65 + i) for i in range(num_dimensions)]  # A, B, C, ...
        
        # Create the generator string based on the resolution
        if resolution == 3:
            # Resolution III Design: Only main effects
            gen = ' '.join(factors)
        elif resolution == 4:
            # Resolution IV Design: Main effects + 2-way interactions
            gen = ' '.join(factors)
            if num_dimensions > 2:
                # Füge erste Interaktionen hinzu (AB, AC)
                gen += ' AB AC'
        else:
            raise ValueError("Resolution must be 3 or 4")
            
        samples = fracfact(gen)
    
    # Scale the values to the actual bounds
    scaled_samples = np.zeros_like(samples, dtype=float)
    for i, (param_name, (min_val, max_val)) in enumerate(bounds.items()):
        scaled_samples[:, i] = (samples[:, i] + 1) / 2 * (max_val - min_val) + min_val
    
    return torch.tensor(scaled_samples, dtype=torch.float64) 