#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Model fitting routines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from itertools import filterfalse
from typing import Any
from warnings import catch_warnings, simplefilter, warn_explicit, WarningMessage

from botorch.exceptions.errors import ModelFittingError, UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.logging import logger
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.closures import get_loss_closure_with_grads
from botorch.optim.core import _LBFGSB_MAXITER_MAXFUN_REGEX
from botorch.optim.fit import fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch.optim.utils import (
    _warning_handler_template,
    get_parameters,
    sample_all_priors,
)
from botorch.utils.context_managers import (
    module_rollback_ctx,
    parameter_rollback_ctx,
    TensorCheckpoint,
)
from botorch.utils.dispatcher import Dispatcher, type_bypassing_encoder
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from linear_operator.utils.errors import NotPSDError
from pyro.infer.mcmc import MCMC, NUTS
from torch import device, Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader


def _debug_warn(w: WarningMessage) -> bool:
    if _LBFGSB_MAXITER_MAXFUN_REGEX.search(str(w.message)):
        return True
    # TODO: Better handle cases where warning handling logic
    # affects both debug and rethrow functions.
    return False


def _rethrow_warn(w: WarningMessage) -> bool:
    if not issubclass(w.category, OptimizationWarning):
        return True
    if "Optimization timed out after" in str(w.message):
        return True
    return False


DEFAULT_WARNING_HANDLER = partial(
    _warning_handler_template,
    debug=_debug_warn,
    rethrow=_rethrow_warn,
)
FitGPyTorchMLL = Dispatcher("fit_gpytorch_mll", encoder=type_bypassing_encoder)


def fit_gpytorch_mll(
    mll: MarginalLogLikelihood,
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    optimizer: Callable | None = None,
    closure_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MarginalLogLikelihood:
    r"""Clearing house for fitting models passed as GPyTorch MarginalLogLikelihoods.

    Args:
        mll: A GPyTorch MarginalLogLikelihood instance.
        closure: Forward-backward closure for obtaining objective values and gradients.
            Responsible for setting parameters' `grad` attributes. If no closure is
            provided, one will be obtained by calling `get_loss_closure_with_grads`.
        optimizer: User specified optimization algorithm. When `optimizer is None`,
            this keyword argument is omitted when calling the dispatcher.
        closure_kwargs: Keyword arguments passed when calling `closure`.
        optimizer_kwargs: A dictionary of keyword arguments passed when
            calling `optimizer`.
        **kwargs: Keyword arguments passed down through the dispatcher to
            fit subroutines. Unexpected keywords are ignored.

    Returns:
        The `mll` instance. If fitting succeeded, then `mll` will be in evaluation mode,
        i.e. `mll.training == False`. Otherwise, `mll` will be in training mode.
    """
    if optimizer is not None:  # defer to per-method defaults
        kwargs["optimizer"] = optimizer

    return FitGPyTorchMLL(
        mll,
        type(mll.likelihood),
        type(mll.model),
        closure=closure,
        closure_kwargs=closure_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        **kwargs,
    )


@FitGPyTorchMLL.register(MarginalLogLikelihood, object, object)
def _fit_fallback(
    mll: MarginalLogLikelihood,
    _: type[object],
    __: type[object],
    *,
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    optimizer: Callable = fit_gpytorch_mll_scipy,
    closure_kwargs: dict[str, Any] | None = None,
    optimizer_kwargs: dict[str, Any] | None = None,
    max_attempts: int = 5,  # Default max_attempts might vary slightly by version
    pick_best_of_all_attempts: bool = False,
    warning_handler: Callable[[WarningMessage], bool] = DEFAULT_WARNING_HANDLER,
    caught_exception_types: tuple[type[BaseException], ...] = (NotPSDError,),
    **ignore: Any,
) -> MarginalLogLikelihood:
    r"""Generic fallback method for fitting Gaussian processes.
    ... (rest of the original docstring) ...
    """
    # Setup
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    params_nograd: dict[str, Parameter] = None  # pyre-ignore [9]
    ckpt_nograd: dict[str, TensorCheckpoint] = None  # pyre-ignore [9]
    ckpt: dict[str, TensorCheckpoint] = None  # pyre-ignore [9]

    # Build closure
    mll.train()
    if closure is None:
        closure = get_loss_closure_with_grads(
            mll, parameters=get_parameters(mll, requires_grad=True)
        )
    if closure_kwargs is not None:
        closure = partial(closure, **closure_kwargs)

    # Record best MLL & corresponding state dict.
    best_mll: float = -float("inf")
    best_state_dict = None
    # Attempt to fit the model
    for attempt in range(1, 1 + max_attempts):
        # Wrap with rollback contextmanager so that each loop iteration reloads the
        # original state_dict upon exiting (unless we clear `ckpt`).
        with module_rollback_ctx(mll, checkpoint=ckpt, device=device("cpu")) as ckpt:
            if attempt > 1:  # resample free parameters
                if params_nograd is None:
                    params_nograd = get_parameters(mll, requires_grad=False)

                if ckpt_nograd is None:  # reuse primary checkpoint
                    ckpt_nograd = {name: ckpt[name] for name in params_nograd}

                with parameter_rollback_ctx(params_nograd, checkpoint=ckpt_nograd):
                    sample_all_priors(mll.model)

            try:
                # Fit the model
                with catch_warnings(record=True) as warning_list:
                    simplefilter("always", category=OptimizationWarning)
                    result = optimizer(mll, closure=closure, **optimizer_kwargs)

                # Resolve warnings and determine whether or not to retry
                success = True
                for w in filterfalse(warning_handler, warning_list):
                    warn_explicit(str(w.message), w.category, w.filename, w.lineno)
                    success = False

                if success and not pick_best_of_all_attempts:
                    # If not picking best of all attempts, return the first
                    # successful attempt.
                    ckpt.clear()  # do not rollback upon exiting
                    return mll.eval()
                elif success:
                    # Update best MLL and corresponding state dict.
                    # Optimizers minimize negative MLL, so we negate fval.
                    current_mll = -result.fval
                    if current_mll > best_mll:
                        best_mll = current_mll
                        # Deepcopy is important here, otherwise they get updated.
                        best_state_dict = deepcopy(mll.state_dict())
                        message = f"Fit attempt #{attempt}: New best MLL: {best_mll}."
                    else:
                        message = (
                            f"Fit attempt #{attempt}: Current MLL {current_mll} did "
                            f"not beat best MLL so far {best_mll}."
                        )
                    logger.debug(message)

                # Ensure mll is in the right mode if going for another attempt.
                mll = mll if mll.training else mll.train()
                if not success:
                    logger.debug(
                        f"Fit attempt #{attempt} of {max_attempts} triggered retry "
                        f"policy {'.' if attempt == max_attempts else '; retrying...'}",
                    )

            except caught_exception_types as err:
                logger.debug(
                    f"Fit attempt #{attempt} of {max_attempts} failed with exception:\n"
                    f"{err}",
                )

    # If picking best of all attempts, return MLL with best state dict.
    if best_state_dict is not None:
        mll.load_state_dict(best_state_dict)
        return mll.eval()

    raise ModelFittingError("All attempts to fit the model have failed.")
    # --- 结束添加日志 ---


@FitGPyTorchMLL.register(SumMarginalLogLikelihood, object, ModelListGP)
def _fit_list(
    mll: SumMarginalLogLikelihood,
    _: type[Likelihood],
    __: type[ModelListGP],
    **kwargs: Any,
) -> SumMarginalLogLikelihood:
    r"""Fitting routine for lists of independent Gaussian processes.

    Args:
        **kwargs: Passed to each of `mll.mlls`.

    Returns:
        The `mll` instance. If fitting succeeded for all of `mll.mlls`, then `mll` will
        be in evaluation mode, i.e. `mll.training == False`. Otherwise, `mll` will be in
        training mode.
    """
    mll.train()
    for sub_mll in mll.mlls:
        fit_gpytorch_mll(sub_mll, **kwargs)

    return mll.eval() if not any(sub_mll.training for sub_mll in mll.mlls) else mll


@FitGPyTorchMLL.register(_ApproximateMarginalLogLikelihood, object, object)
def _fit_fallback_approximate(
    mll: _ApproximateMarginalLogLikelihood,
    _: type[Likelihood],
    __: type[ApproximateGPyTorchModel],
    *,
    closure: Callable[[], tuple[Tensor, Sequence[Tensor | None]]] | None = None,
    data_loader: DataLoader | None = None,
    optimizer: Callable | None = None,
    full_batch_limit: int = 1024,
    **kwargs: Any,
) -> _ApproximateMarginalLogLikelihood:
    r"""Fallback method for fitting approximate Gaussian processes.

    Args:
        closure: Forward-backward closure for obtaining objective values and gradients.
            Responsible for setting parameters' `grad` attributes. If no closure is
            provided, one will be obtained by calling `get_loss_closure_with_grads`.
        optimizer: The underlying optimization algorithm to run. Default to
            `fit_gpytorch_mll_scipy` when `closure=None` and the model's internal
            training set has no more than `full_batch_cutoff` observations; otherwise,
            defaults to `fit_gpytorch_mll_torch`.
        data_loader: An optional DataLoader to pass to `get_loss_closure_with_grads`.
            May only be provided when `closure=None`.
        full_batch_limit: Threshold for determining the default choice of `optimizer`
            when `closure=None`.
        **kwargs: Keyword arguments passed to `_fit_fallback`.
    """
    if data_loader is not None:
        if closure is not None:
            raise UnsupportedError(
                "Only one of `data_loader` or `closure` may be passed."
            )
        closure = get_loss_closure_with_grads(
            mll=mll,
            data_loader=data_loader,
            parameters=get_parameters(mll, requires_grad=True),
        )

    if optimizer is None:
        optimizer = (
            fit_gpytorch_mll_scipy
            if closure is None and len(mll.model.train_targets) <= full_batch_limit
            else fit_gpytorch_mll_torch
        )

    return _fit_fallback(mll, _, __, closure=closure, optimizer=optimizer, **kwargs)


def fit_fully_bayesian_model_nuts(
    model: SaasFullyBayesianSingleTaskGP | SaasFullyBayesianMultiTaskGP,
    max_tree_depth: int = 6,
    warmup_steps: int = 512,
    num_samples: int = 256,
    thinning: int = 16,
    disable_progbar: bool = False,
    jit_compile: bool = False,
) -> None:
    r"""Fit a fully Bayesian model using the No-U-Turn-Sampler (NUTS)


    Args:
        model: SaasFullyBayesianSingleTaskGP to be fitted.
        max_tree_depth: Maximum tree depth for NUTS
        warmup_steps: The number of burn-in steps for NUTS.
        num_samples:  The number of MCMC samples. Note that with thinning,
            num_samples / thinning samples are retained.
        thinning: The amount of thinning. Every nth sample is retained.
        disable_progbar: A boolean indicating whether to print the progress
            bar and diagnostics during MCMC.
        jit_compile: Whether to use jit. Using jit may be ~2X faster (rough estimate),
            but it will also increase the memory usage and sometimes result in runtime
            errors, e.g., https://github.com/pyro-ppl/pyro/issues/3136.

    Example:
        >>> gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(gp)
    """
    model.train()

    # Do inference with NUTS
    nuts = NUTS(
        model.pyro_model.sample,
        jit_compile=jit_compile,
        full_mass=True,
        ignore_jit_warnings=True,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        nuts,
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        disable_progbar=disable_progbar,
    )
    mcmc.run()

    # Get final MCMC samples from the Pyro model
    mcmc_samples = model.pyro_model.postprocess_mcmc_samples(
        mcmc_samples=mcmc.get_samples()
    )
    for k, v in mcmc_samples.items():
        mcmc_samples[k] = v[::thinning]

    # Load the MCMC samples back into the BoTorch model
    model.load_mcmc_samples(mcmc_samples)
    model.eval()
