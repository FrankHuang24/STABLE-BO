import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.kernels import Kernel

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

def get_next_points(acq, kernel, n_mixture, init_x, init_y, best_init_y, bounds, n_points=1, n_mixture1=1, n_mixture2=1):
    noise_prior = GammaPrior(1.1, 0.5)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=[],
        noise_constraint=GreaterThan(
            0.0001,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )
    single_model = SingleTaskGP(init_x, init_y, likelihood=likelihood, covar_module=kernel)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)
    if acq == 'ei':
        acq_function = ExpectedImprovement(model=single_model, best_f=best_init_y, maximize=False)
    elif acq == 'pi':
        acq_function = ProbabilityOfImprovement(model=single_model, best_f=best_init_y, maximize=False)
    elif acq == 'ucb':
        acq_function = UpperConfidenceBound(model=single_model, beta=0.5, maximize=False)
    else:
        raise ValueError('Acquisition function not identified')
    candidates, _ = optimize_acqf(acq_function=acq_function, bounds=bounds, q=n_points, num_restarts=100,
                                  raw_samples=512, options={"batch_limit": 5, "maxiter": 100})
    return candidates

class StableKernel(Kernel):
    def __init__(self, r=2.0, lengthscale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        lengthscale = torch.tensor(lengthscale).view(1, 1)
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(lengthscale))
    
    @property
    def lengthscale(self):
        return self.raw_lengthscale

    def forward(self, x1, x2, **params):
        diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
        dist = diff.pow(self.r).sum(-1)
        return torch.exp(-dist / self.lengthscale)

class MixedStableKernel(Kernel):
    def __init__(self, r_list, weights=None, lengthscale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.r_list = r_list
        if weights is None:
            weights = [1.0 / len(r_list)] * len(r_list)
        self.weights = weights
        lengthscale = torch.tensor(lengthscale).view(1, 1)
        self.register_parameter(name="raw_lengthscale", parameter=torch.nn.Parameter(lengthscale))
    
    @property
    def lengthscale(self):
        return self.raw_lengthscale

    def forward(self, x1, x2, **params):
        result = 0
        for w, r in zip(self.weights, self.r_list):
            diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
            dist = diff.pow(r).sum(-1)
            result += w * torch.exp(-dist / self.lengthscale)
        return result