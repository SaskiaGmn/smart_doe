import numpy as np
import torch
from typing import Dict, Tuple
from pyDOE3 import fracfact, ff2n, lhs  # Nutze pyDOE2 statt pyDOE
import itertools
#Example bounds
bounds: Dict[str, Tuple[float, float]] = {
    "x1": (0.0, 1.0),
    "x2": (10.0, 20.0),
    "x3": (100.0, 200.0),
    "x4": (0.0, 5.0),
    "x5": (2, 6),
}

# Number of dimensions
num_dimensions = len(bounds)
main_factors = 3
total_factors = 5
# Check if design is possible
max_possible_interactions = sum(
    len(list(itertools.combinations(range(main_factors), r)))
    for r in range(2, main_factors + 1)
)


num_generated = total_factors - main_factors
if max_possible_interactions < num_generated:
    raise ValueError("Design is not possible with the given number of main factors")

# Create aliases A, B, C, D, ...
factors = [chr(65 + i) for i in range(total_factors)]  # ['A', 'B', 'C', 'D']
if main_factors >= total_factors:
    raise ValueError("Number of main factors must be less than total factors")

    # Name of main factors: A, B, C, ...
letters = [chr(65 + i) for i in range(main_factors)]  # 'A', 'B', ...

    # Initial generator: main factors as own columns
generator = list(letters)

    # Number of columns to generate
num_generated = total_factors - main_factors

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
print(samples)

#Comment out for space filling lhs
'''num_points = 15
samples = lhs(num_dimensions, samples=num_points, criterion='maximin')
scaled_samples = np.zeros_like(samples)
for i, (param_name, (min_val, max_val)) in enumerate(bounds.items()):
    scaled_samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
'''
#Comment out for full factorial design
'''samples = ff2n(num_dimensions)'''

# Comment out for fractional factorial design
'''samples = fracfact(gen)
gen = ' '.join(factors)'''

# Scale
'''scaled_samples = np.zeros_like(samples, dtype=float)
for i, (param_name, (min_val, max_val)) in enumerate(bounds.items()):
    scaled_samples[:, i] = (samples[:, i] + 1) / 2 * (max_val - min_val) + min_val

# Konvertiere zu Torch-Tensor
tensor_samples = torch.tensor(scaled_samples, dtype=torch.float64)

# Drucke die DoE-Tabelle mit Parameter-Namen
header = list(bounds.keys())
print("\t".join(header))
for row in tensor_samples.numpy():
    print("\t".join(f"{val:.2f}" for val in row))'''
