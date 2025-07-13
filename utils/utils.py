# utils.py

import plotly.graph_objects as go
import torch
import numpy as np
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.kernels import Kernel
from gpytorch.module import Module

from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, qKnowledgeGradient
from botorch.optim import optimize_acqf
from typing import Union, List, Optional

def get_next_points(
    acq: str,
    kernel: Union[str, Module], 
    n_mixture: int,
    init_x: torch.Tensor,
    init_y: torch.Tensor,
    best_init_y: float,
    bounds: torch.Tensor,
    n_points: int = 1,
    n_mixture1: int = 1,
    n_mixture2: int = 1,
    # --- 新增：添加 alphas 参数 ---
    alphas: Optional[List[float]] = None
    # --- 结束新增 ---
) -> torch.Tensor:
    """
    Fits a GP model and computes the next acquisition points.

    Args:
        acq: The acquisition function choice ('ei', 'pi', 'ucb').
        kernel: The kernel to use. Can be a string identifier or an instantiated
                GPyTorch kernel Module.
        n_mixture: Number of mixtures (used if kernel is 'gsm' or 'csm' string).
        init_x: Training inputs.
        init_y: Training outputs.
        best_init_y: The best observed objective value so far.
        bounds: The parameter bounds.
        n_points: The number of points to acquire.
        n_mixture1: Number of Cauchy mixtures (used if kernel is 'mix' string).
        n_mixture2: Number of Gaussian mixtures (used if kernel is 'mix' string).
        alphas: List of fixed alpha values for 'mixstable_fixed_alpha' kernel.

    Returns:
        The next candidate points.
    """
    noise_prior = GammaPrior(1.1, 0.5)
    noise_constraint = GreaterThan(1e-4)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=[],
        noise_constraint=GreaterThan(
            1e-4,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )

    # Instantiate the SingleTaskGP model.
    single_model = SingleTaskGP(
        train_X=init_x,
        train_Y=init_y,
        likelihood=likelihood,
        covar_module=kernel,       # 传递核字符串或对象
        n_mixture=n_mixture,
        n_mixture1=n_mixture1,
        n_mixture2=n_mixture2,
        alphas=alphas              
    )
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)

    # --- 保持使用多次重启和选择最优的 MLL 拟合 ---
    try:
        optimizer_kwargs = {'options': {'maxiter': 200}}
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs=optimizer_kwargs
        )
    except Exception as fit_error:
        print(f"Error during MLL fitting: {fit_error}")
        single_model.eval()


    # Select acquisition function
    if acq == 'ei':
        acq_function = ExpectedImprovement(model=single_model, best_f=best_init_y, maximize=False)
    elif acq == 'pi':
        acq_function = ProbabilityOfImprovement(model=single_model, best_f=best_init_y, maximize=False)
    elif acq == 'ucb':
        acq_function = UpperConfidenceBound(model=single_model, beta=0.2, maximize=False)
    else:
        raise ValueError(f"Acquisition function '{acq}' not identified")

    # Optimize acquisition function
    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=n_points,
        num_restarts=10, 
        raw_samples=512, 
        options={"batch_limit": 5, "maxiter": 200} 
    )
    print("--- Learned Kernel Hyperparameters ---")
    covar_module = single_model.covar_module
    if hasattr(covar_module, 'alphas'): # 检查是否是我们的可学习 alpha 核
        # alpha 的形状是 batch x Q
        print(f"Alphas: {covar_module.alphas.detach().cpu().numpy()}")
    if hasattr(covar_module, 'mixture_weights'):
        print(f"Weights: {covar_module.mixture_weights.detach().cpu().numpy()}")
    if hasattr(covar_module, 'mixture_means'):
        # 均值形状是 batch x Q x 1 x d
        print(f"Means (gamma): {covar_module.mixture_means.detach().cpu().numpy()}")
    if hasattr(covar_module, 'mixture_scales'):
        # 尺度形状是 batch x Q x 1 x d
        print(f"Scales (delta): {covar_module.mixture_scales.detach().cpu().numpy()}")
    if hasattr(single_model.likelihood, 'noise'):
        print(f"Likelihood Noise: {single_model.likelihood.noise.detach().cpu().numpy()}")
    print("------------------------------------")

    return candidates

def compute_acquisition_function(single_model, best_init_y, l_bound=-2., h_bound=10., resolution=1000):
    linspace = torch.linspace(l_bound, h_bound, steps=resolution, dtype=single_model.train_inputs[0].dtype, device=single_model.train_inputs[0].device)
    EI = ExpectedImprovement(model=single_model, best_f=best_init_y, maximize=False) # Ensure maximize=False
    result = []
    with torch.no_grad():
        acq_values = EI(linspace.unsqueeze(-1).unsqueeze(-1)) # Shape needs to match model input [q x n_eval x d] -> [res x 1 x d]
    return acq_values # Return tensor directly


def print_acquisition_function(acq_fun, iteration, l_bound=-2., h_bound=10., resolution=1000, suggested=None):
    x_plot = torch.linspace(l_bound, h_bound, steps=resolution).numpy()
    z_plot = acq_fun.cpu().detach().numpy() # Ensure tensor is moved to cpu for numpy conversion
    max_acq_fun_idx = acq_fun.argmax().item()
    max_acq_fun_x = x_plot[max_acq_fun_idx]

    data = go.Scatter(x=x_plot, y=z_plot, line_color="yellow", name='Acquisition Function')

    fig = go.Figure(data=data)
    fig.update_layout(title=f"Acquisition function. Iteration {iteration}",
                      xaxis_title="input", yaxis_title="Acquisition Value")
    if suggested is not None:
        fig.add_vline(x=float(suggested[0][0]), line_width=3, line_dash="dash", line_color="red", name='Suggested Point')
        fig.add_vline(x=max_acq_fun_x, line_width=1, line_dash="dot", line_color="orange", name='Max Acq Value X')
    else:
        # Mark the maximum value if no specific suggestion is passed
         fig.add_vline(x=max_acq_fun_x, line_width=3, line_color="red", name='Max Acq Value X')
    fig.show()


def compute_predictive_distribution(single_model, l_bound=-2., h_bound=10., resolution=1000):
    linspace = torch.linspace(l_bound, h_bound, steps=resolution, dtype=single_model.train_inputs[0].dtype, device=single_model.train_inputs[0].device)
    # Ensure correct input shape for model: [resolution x 1 x d]
    x_test = linspace.unsqueeze(-1).unsqueeze(-1)
    with torch.no_grad():
         posterior = single_model.posterior(x_test)
         mean = posterior.mean.squeeze(-1).squeeze(-1) # Squeeze output dimensions
         variance = posterior.variance.squeeze(-1).squeeze(-1) # Squeeze output dimensions
    return mean, variance # Return tensors directly


def print_predictive_mean(predictive_mean, predictive_variance, iteration, l_bound=-2., h_bound=10., resolution=1000,
                          suggested=None, old_obs=[], old_values=[]):
    x_plot = torch.linspace(l_bound, h_bound, steps=resolution).numpy()
    mean_plot = predictive_mean.cpu().detach().numpy()
    variance_plot = predictive_variance.cpu().detach().numpy()
    std_dev = np.sqrt(variance_plot)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_plot, y=mean_plot + 1.96 * std_dev, 
                             mode='lines',
                             line=dict(color="#19D3F3", width=0.1),
                             showlegend=False)) # No need for legend for bounds usually
    fig.add_trace(go.Scatter(x=x_plot, y=mean_plot,
                             mode='lines',
                             line=dict(color="blue"),
                             fill='tonexty', # Fill between this trace and the previous one (upper bound)
                             name='Predictive Mean & 95% CI'))
    fig.add_trace(go.Scatter(x=x_plot, y=mean_plot - 1.96 * std_dev,
                             mode='lines',
                             line=dict(color="blue", width=0.1),
                             fill='tonexty', # Fill between this trace and the previous one (mean)
                             showlegend=False)) 

    # Add observed data points
    if len(old_obs) > 0:
        old_obs_np = old_obs.cpu().numpy().flatten() 
        old_values_np = old_values.cpu().numpy().flatten()
        fig.add_trace(go.Scatter(x=old_obs_np, y=old_values_np, mode='markers',
                                 marker=dict(color="black", size=8), name='Observations'))

    # Add suggested point
    if suggested is not None:
        fig.add_vline(x=float(suggested[0][0]), line_width=3, line_dash="dash", line_color="red", name='Suggested Point')

    fig.update_layout(title=f"GP Predictive distribution. Iteration {iteration}", xaxis_title="input",
                      yaxis_title="output", showlegend=True)

    fig.show()


def print_objective_function(target_function, best_candidate, iteration, l_bound=-2., h_bound=10., resolution=100):
    x_plot = np.linspace(l_bound, h_bound, resolution)
    # Assuming target_function can handle numpy array or needs tensor
    try:
         z_plot = target_function(torch.tensor(x_plot.reshape(-1, 1), dtype=torch.float32)).cpu().numpy().flatten()
    except:
         z_plot = target_function(x_plot.reshape(-1, 1)).flatten()

    data = go.Scatter(x=x_plot, y=z_plot, line_color="#FE73FF", name='True Objective')
    fig = go.Figure(data=data)
    fig.update_layout(title=f"Objective function. Iteration {iteration}", xaxis_title="input",
                      yaxis_title="output")
    # best_candidate might be tensor, ensure it's float
    if best_candidate is not None:
         fig.add_vline(x=float(best_candidate), line_width=3, line_dash="dot", line_color="green", name='Best Found X')
    fig.show()


def visualize_functions(target_function, single_model, best_init_y, best_candidate, candidate_acq_fun, iteration,
                        previous_observations, previous_values, l_bound=-2., h_bound=10.):
    resolution = 500 # Use a decent resolution for plots
    predictive_mean, predictive_variance = compute_predictive_distribution(single_model, l_bound=l_bound, h_bound=h_bound, resolution=resolution)
    print_predictive_mean(predictive_mean, predictive_variance, iteration, suggested=candidate_acq_fun,
                          old_obs=previous_observations, old_values=previous_values, l_bound=l_bound, h_bound=h_bound, resolution=resolution)
    acq_fun = compute_acquisition_function(single_model, best_init_y, l_bound=l_bound, h_bound=h_bound, resolution=resolution)
    print_acquisition_function(acq_fun, iteration, suggested=candidate_acq_fun, l_bound=l_bound, h_bound=h_bound, resolution=resolution)
    print_objective_function(target_function, best_candidate, iteration, l_bound=l_bound, h_bound=h_bound, resolution=100)


def get_next_points_and_visualize(target_function, init_x, init_y, best_init_y, bounds, iteration,
                                  previous_observations,
                                  previous_values, n_points=1, kernel='matern'): # Added kernel arg, defaulting to matern
    # NOTE: This function might need updates to handle custom kernels like get_next_points if used extensively.
    # For now, it defaults to SingleTaskGP's default (likely RBF or Matern depending on BoTorch version/its internal logic)
    # or accepts a kernel string.
    single_model = SingleTaskGP(
        train_X=init_x,
        train_Y=init_y,
        )
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)

    EI = ExpectedImprovement(model=single_model, best_f=best_init_y, maximize=False) # Ensure maximize=False

    candidates, _ = optimize_acqf(
        acq_function=EI,
        bounds=bounds,
        q=n_points,
        num_restarts=10, 
        raw_samples=256, 
        options={"batch_limit": 5, "maxiter": 100}
    )
    
    best_candidate_idx = (init_y == best_init_y).nonzero(as_tuple=True)[0]
    if len(best_candidate_idx) > 0:
        best_candidate = init_x[best_candidate_idx[0]][0].item() # Assuming 1D input for visualization
    else:
        best_candidate = None 

    # Determine plot bounds based on data or default
    l_bound = bounds[0][0].item() if bounds is not None else -2.
    h_bound = bounds[1][0].item() if bounds is not None else 10.

    visualize_functions(
        target_function, single_model, best_init_y, best_candidate, candidates, iteration,
        previous_observations, previous_values, l_bound=l_bound, h_bound=h_bound
    )

    return candidates

