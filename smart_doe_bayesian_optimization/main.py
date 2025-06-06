import models
import torch
from models.gp_model import BaseGPModel
from models.kernel_factory import KernelFactory
from gpytorch.priors import NormalPrior
from models.likelihood_factory import LikelihoodFactory
from data.create_dataset import DatasetManager
from data.function_factory import FunctionFactory
from gpytorch.constraints import GreaterThan
from models.optimizer_factory import OptimizerFactory
from optimization.optimization import GPOptimizer
import matplotlib.pyplot as plt
from gpytorch.priors.torch_priors import GammaPrior

from botorch import fit_gpytorch_mll

# Create a multi-dimensional dataset
dataset = DatasetManager(dtype=torch.float64, num_input_dimensions=3)
dataset.func_create_dataset(
    FunctionFactory.multi_inputs,
    num_datapoints=5,
    sampling_method="random",
    noise_level=0.1,
    x1_range=(0, 10),
    x2_range=(0, 10),
    x3_range=(0, 10)
)

# Erstelle einen Matern Kernel mit ARD für mehrere Dimensionen
kernel = KernelFactory.create_kernel(
    'Matern',
    nu=2.5,
    lengthscale_prior=GammaPrior(3.0, 6.0)
)

# Erstelle die Likelihood
likelihood = LikelihoodFactory.create_likelihood(
    'Gaussian',
    noise_constraint=GreaterThan(1e-5)
)

# Definiere die Skalierung
scaling_dict = {
    'input': 'normalize',
    'output': 'standardize'
}

# Erstelle und trainiere das GP-Modell
gp_model = BaseGPModel(
    "SingleTaskGP",
    "ExactMarginalLogLikelihood",
    "adam",
    kernel,
    dataset.unscaled_data[0],
    dataset.unscaled_data[1],
    likelihood,
    bounds_list=dataset.bounds_list,
    scaling_dict=scaling_dict,
    optimizer_kwargs={"lr": 0.1}
)

gp_model.train(num_epochs=100)

# Create the optimizer
gp_optimizer = GPOptimizer(
    base_model=gp_model,
    acq_func_type="LogExp_Improvement",
    is_maximization=True
)

print("\nStart der Optimierung...")
print("Bei jedem Schritt werden Sie nach dem beobachteten Wert gefragt.")
print("Geben Sie bitte einen numerischen Wert ein.\n")

# Perform the optimization
gp_optimizer.optimization_loop(
    num_restarts=40,
    raw_samples=400,
    max_iterations=5
)