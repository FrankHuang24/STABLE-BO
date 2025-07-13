#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gaussian Process Regression models based on GPyTorch models.

These models are often a good starting point and are further documented in the
tutorials.

`SingleTaskGP` is a single-task exact GP model that uses relatively strong priors on
the Kernel hyperparameters, which work best when covariates are normalized to the unit
cube and outcomes are standardized (zero mean, unit variance). By default, this model
uses a `Standardize` outcome transform, which applies this standardization. However,
it does not (yet) use an input transform by default.

`SingleTaskGP` model works in batch mode (each batch having its own hyperparameters).
When the training observations include multiple outputs, `SingleTaskGP` uses
batching to model outputs independently.

`SingleTaskGP` supports multiple outputs. However, as a single-task model,
`SingleTaskGP` should be used only when the outputs are independent and all
use the same training inputs. If outputs are independent but they have different
training inputs, use the `ModelListGP`. When modeling correlations between outputs,
use a multi-task model like `MultiTaskGP`.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union, List, Any


import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils import validate_input_scaling
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior, 
    get_gaussian_likelihood_with_lognormal_prior,
)
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.module import Module # Import Module for isinstance check
# Import standard kernels needed for string handling
from gpytorch.kernels import (
    Kernel,
    MaternKernel,
    RBFKernel,
    RQKernel,
    SpectralDeltaKernel,
    SpectralMixtureKernel,
    PeriodicKernel,
    LinearKernel
)

try:
    from botorch.models.kernels.adaptive_kernel import AdaptiveKernel
except ImportError:
    AdaptiveKernel = None 
    warnings.warn("Could not import AdaptiveKernel. 'ada' kernel type will not be available.")

# --- MODIFICATION: Import custom kernels ---
try:
    # --- MODIFICATION: Import custom kernels ---
    from .kernels.stable_kernel import MixedFixedAlphaStableKernel, LearnableAlphaStableKernel
except ImportError:
    try:
        from kernels.stable_kernel import MixedFixedAlphaStableKernel, LearnableAlphaStableKernel
    except ImportError:
        MixedFixedAlphaStableKernel = None
        LearnableAlphaStableKernel = None # 也要为新核设置回退
        warnings.warn("Could not import stable_kernel. Custom stable kernels will not be available.")

try:
    from .kernels.cauchy_spectral_mixture import CauchyMixtureKernel
except ImportError:
    try:
        from kernels.cauchy_spectral_mixture import CauchyMixtureKernel
    except ImportError:
         CauchyMixtureKernel = None 
         warnings.warn("Could not import CauchyMixtureKernel. 'csm' or 'mix' kernel types might not be available.")

# --- END MODIFICATION ---

from gpytorch.models.exact_gp import ExactGP
from torch import Tensor


class SingleTaskGP(BatchedMultiOutputGPyTorchModel, ExactGP, FantasizeMixin):
    r"""A single-task exact GP model, supporting both known and inferred noise levels.
    ... (类的文档字符串保持不变) ...
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Union[Module, str, None] = None,
        n_mixture: Optional[int] = None, # Kept for gsm/csm string init
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        n_mixture1: Optional[int] = None, # Kept for mix string init
        n_mixture2: Optional[int] = None, # Kept for mix string init
        # --- 新增 alphas 参数 ---
        alphas: Optional[List[float]] = None,
        # --- 结束新增 ---
    ) -> None:
        r"""
        Args:
            # ... (其他参数的文档字符串) ...
            alphas: List of fixed alpha values for 'mixstable_fixed_alpha' kernel.
        """
        self._original_train_X = train_X
        self._original_train_Y = train_Y

        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        if outcome_transform == DEFAULT:
            outcome_transform = Standardize(
                m=train_Y.shape[-1], batch_shape=train_X.shape[:-2]
            )

        # Apply input transform before setting dimensions and other checks
        self._input_transform_applied = False
        if input_transform is not None:
            if not isinstance(input_transform, Module):
                 input_transform = torch.nn.ModuleList([input_transform])
            self.input_transform = input_transform
            # 使用 transform_inputs 处理训练数据转换
            transformed_X = self.transform_inputs(X=train_X)
            self._input_transform_applied = True
        else:
            transformed_X = train_X

        # Apply outcome transform
        # Store transformed targets for GP initialization
        if outcome_transform is not None:
            train_Y_t, train_Yvar_t = outcome_transform(train_Y, train_Yvar)
        else:
            train_Y_t, train_Yvar_t = train_Y, train_Yvar


        # Validate tensor args again after transforms
        self._validate_tensor_args(X=transformed_X, Y=train_Y_t, Yvar=train_Yvar_t)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X,
            train_Y=train_Y_t,
            train_Yvar=train_Yvar_t,
            ignore_X_dims=ignore_X_dims,
        )

        # Set dimensions based on original train_X shape
        self._set_dimensions(train_X=train_X, train_Y=train_Y)

        # Transform tensor args for GPyTorch model init (uses transformed_X and transformed Y)
        t_train_X, t_train_Y, t_train_Yvar = self._transform_tensor_args(
            X=transformed_X, Y=train_Y_t, Yvar=train_Yvar_t
        )

        # Determine likelihood
        if likelihood is None:
            if train_Yvar is None: # Check original Yvar
                likelihood = get_gaussian_likelihood_with_lognormal_prior(
                    batch_shape=self._aug_batch_shape
                )
            else:
                likelihood = FixedNoiseGaussianLikelihood(
                    noise=t_train_Yvar, batch_shape=self._aug_batch_shape # Use transformed Yvar
                )
        else:
            self._is_custom_likelihood = True

        # --- Kernel Handling Logic ---
        ard_num_dims = transformed_X.shape[-1] # Use dimension of (potentially) transformed data
        batch_shape = self._aug_batch_shape
        covar: Union[Module, None] = None # Initialize covar

        if isinstance(covar_module, Module):
            covar = covar_module
            print(f"Using provided kernel module instance: {type(covar)}")
        elif isinstance(covar_module, str):
            print(f"Interpreting kernel string: {covar_module}")
            if covar_module == "rbf":
                 covar = RBFKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
            elif covar_module == 'matern':
                 covar = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
            elif covar_module == 'rq':
                 covar = RQKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
            elif covar_module == 'pe':
                 covar = PeriodicKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
            elif covar_module == 'gsm':
                 if n_mixture is None: raise ValueError("n_mixture required for gsm kernel specified as string")
                 covar = SpectralMixtureKernel(num_mixtures=n_mixture, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
            elif covar_module == 'csm':
                 if CauchyMixtureKernel is None: raise ImportError("CauchyMixtureKernel not available.")
                 if n_mixture is None: raise ValueError("n_mixture required for csm kernel specified as string")
                 covar = CauchyMixtureKernel(num_mixtures=n_mixture, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
            elif covar_module == 'mix':
                 if CauchyMixtureKernel is None: raise ImportError("CauchyMixtureKernel not available.")
                 if n_mixture1 is None or n_mixture2 is None: raise ValueError("n_mixture1 and n_mixture2 required for mix kernel specified as string")
                 covar1 = CauchyMixtureKernel(num_mixtures=n_mixture1, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
                 covar2 = SpectralMixtureKernel(num_mixtures=n_mixture2, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
                 covar = covar1 + covar2
            elif covar_module == 'sdk':
                 covar = SpectralDeltaKernel(num_dims=ard_num_dims, batch_shape=batch_shape)
            # REMOVED: SincKernel handling
            elif covar_module == 'ada':
                 if AdaptiveKernel is None: raise ImportError("AdaptiveKernel not available.")
                 kernel_list = [
                    RBFKernel(batch_shape=batch_shape),
                    MaternKernel(nu=2.5, batch_shape=batch_shape),
                    LinearKernel(batch_shape=batch_shape),
                    RQKernel(batch_shape=batch_shape)
                 ]
                 covar = AdaptiveKernel(kernel_list, batch_shape=batch_shape)
            # --- 新增：处理 mixstable_fixed_alpha ---
            elif covar_module == 'mixstable_fixed_alpha':
                 if MixedFixedAlphaStableKernel is None: raise ImportError("MixedFixedAlphaStableKernel not available.")
                 if alphas is None:
                     raise ValueError("`alphas` must be provided for 'mixstable_fixed_alpha' kernel")
                 covar = MixedFixedAlphaStableKernel(
                     alphas=alphas,
                     num_dims=ard_num_dims, # 使用 ard_num_dims
                     batch_shape=batch_shape
                 )
                 print(f"Instantiated MixedFixedAlphaStableKernel with alphas: {alphas}")
            elif covar_module == 'learnable_alpha_stable':
                 if LearnableAlphaStableKernel is None: raise ImportError("LearnableAlphaStableKernel not available.")
                 if n_mixture is None: # 现在 n_mixture 是必需的，因为它定义了Q
                     raise ValueError("`n_mixture` (Q) must be provided for 'learnable_alpha_stable' kernel")

                 covar = LearnableAlphaStableKernel(
                     num_mixtures=n_mixture, # 使用 n_mixture 作为 Q
                     num_dims=ard_num_dims,
                     batch_shape=batch_shape
                 )
                 print(f"Instantiated LearnableAlphaStableKernel with Q={n_mixture} components.")
            
            else:
                 warnings.warn(f"Unknown kernel string '{covar_module}'. Defaulting to kernel returned by get_covar_module_with_dim_scaled_prior.", UserWarning)
                 covar = get_covar_module_with_dim_scaled_prior(
                     ard_num_dims=ard_num_dims, batch_shape=batch_shape
                 )
        elif covar_module is None:
             print("covar_module is None. Using default kernel from get_covar_module_with_dim_scaled_prior.")
             covar = get_covar_module_with_dim_scaled_prior(
                 ard_num_dims=ard_num_dims, batch_shape=batch_shape
             )
        else:
            raise TypeError(...)

        # Initialize ExactGP
        ExactGP.__init__(self, train_inputs=t_train_X, train_targets=t_train_Y, likelihood=likelihood)

        # Set mean and covariance modules
        if mean_module is None:
            mean_module = ConstantMean(batch_shape=batch_shape)
        self.mean_module = mean_module
        self.covar_module = covar

        # --- 调用基于数据的初始化 (优先经验谱，然后统计量) ---
        initialized_successfully = False
        if hasattr(self.covar_module, "initialize_from_data_empspect"):
            try:
                print(f"Attempting empirical spectrum initialization for {type(self.covar_module).__name__}...")
                self.covar_module.initialize_from_data_empspect(self._original_train_X, self._original_train_Y)
                print(f"Kernel {type(self.covar_module).__name__} initialized from empirical spectrum.")
                initialized_successfully = True
            except Exception as emp_spec_e:
                print(f"Warning: Empirical spectrum initialization failed for kernel "
                      f"{type(self.covar_module).__name__}: {emp_spec_e}")
                # 如果经验谱失败，尝试基于统计量的初始化（如果存在）

        if not initialized_successfully and hasattr(self.covar_module, "initialize_from_data"):
            try:
                print(f"Falling back to data statistics initialization for {type(self.covar_module).__name__}...")
                self.covar_module.initialize_from_data(self._original_train_X, self._original_train_Y)
                print(f"Kernel {type(self.covar_module).__name__} initialized from data statistics.")
                initialized_successfully = True
            except Exception as stat_e:
                print(f"Warning: Data statistics initialization also failed for kernel "
                      f"{type(self.covar_module).__name__}: {stat_e}")

        if not initialized_successfully:
            print(f"Warning: Could not initialize kernel {type(self.covar_module).__name__} from data. "
                  "Kernel will use its default/random parameter initialization (if any).")

        # Set subset batch dict (尝试通用化)
        self._subset_batch_dict = {"mean_module.raw_constant": -1}
        if train_Yvar is None and hasattr(self.likelihood, "noise_covar"):
             self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        if hasattr(self.covar_module, "raw_outputscale"):
             self._subset_batch_dict["covar_module.raw_outputscale"] = -1
        if hasattr(self.covar_module, "base_kernel") and hasattr(self.covar_module.base_kernel, "raw_lengthscale"):
            try:
                 lengthscale_shape = getattr(self.covar_module.base_kernel, "raw_lengthscale", torch.empty(0)).shape
                 lengthscale_batch_dim = -3 if len(lengthscale_shape) >= 3 else -1
                 self._subset_batch_dict["covar_module.base_kernel.raw_lengthscale"] = lengthscale_batch_dim
            except Exception:
                 self._subset_batch_dict["covar_module.base_kernel.raw_lengthscale"] = -1
        elif hasattr(self.covar_module, "raw_lengthscale"):
            try:
                 lengthscale_shape = getattr(self.covar_module, "raw_lengthscale", torch.empty(0)).shape
                 lengthscale_batch_dim = -3 if len(lengthscale_shape) >= 3 else -1
                 self._subset_batch_dict["covar_module.raw_lengthscale"] = lengthscale_batch_dim
            except Exception:
                 self._subset_batch_dict["covar_module.raw_lengthscale"] = -1

        # Set transforms
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        # Move model to correct device/dtype (使用原始 train_X)
        self.to(self._original_train_X)

    @classmethod
    def construct_inputs(
        cls, training_data: SupervisedDataset, **kwargs
    ) -> dict[str, BotorchContainer | Tensor]:
        return super().construct_inputs(training_data=training_data, **kwargs)


    def forward(self, x: Tensor) -> MultivariateNormal:
        processed_x = x
        # Only apply transform if training, otherwise it's handled in posterior
        if self.training and hasattr(self, "input_transform"):
             processed_x = self.transform_inputs(x)

        mean_x = self.mean_module(processed_x)
        covar_x = self.covar_module(processed_x)
        return MultivariateNormal(mean_x, covar_x)
