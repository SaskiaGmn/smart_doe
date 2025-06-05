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
        # Für 2 oder weniger Faktoren verwenden wir ein vollständiges 2^k Design
        samples = ff2n(num_dimensions)
    else:
        # Für mehr Faktoren verwenden wir ein fraktionales Design
        # Erstelle die Generator-String basierend auf der Auflösung
        if resolution == 3:
            # Resolution III Design
            gen = ' '.join([f'x{i+1}' for i in range(num_dimensions)])
        elif resolution == 4:
            # Resolution IV Design
            gen = ' '.join([f'x{i+1}' for i in range(num_dimensions)])
            # Füge Generatoren für Resolution IV hinzu
            gen += ' x1*x2 x1*x3'
        else:
            raise ValueError("Resolution must be 3 or 4")
            
        samples = fracfact(gen)
    
    # Skaliere die Werte auf die tatsächlichen Grenzen
    scaled_samples = np.zeros_like(samples, dtype=float)
    for i, (param_name, (min_val, max_val)) in enumerate(bounds.items()):
        scaled_samples[:, i] = (samples[:, i] + 1) / 2 * (max_val - min_val) + min_val
    
    return torch.tensor(scaled_samples, dtype=torch.float64) 