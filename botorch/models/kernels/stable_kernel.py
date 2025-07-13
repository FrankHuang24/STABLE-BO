import math
from typing import List, Optional, Union
import numpy as np
from scipy.fftpack import fft
import torch
from gpytorch.constraints import Interval, Positive,GreaterThan
from gpytorch.priors import Prior, LogNormalPrior, NormalPrior, GammaPrior
from gpytorch.kernels.kernel import Kernel

class MixedFixedAlphaStableKernel(Kernel):
    r"""
    混合稳定核函数，多个具有不同稳定参数 (alpha) 的核。
    """

    is_stationary = True
    has_lengthscale = False

    # __init__ remains largely the same, EXCEPT we remove the call
    # to self.initialize_params() at the end.
    def __init__(
        self,
        alphas: List[float], 
        num_dims: int,     
        batch_shape: Optional[torch.Size] = torch.Size([]),
        mixture_weights_prior: Optional[Prior] = None,
        mixture_weights_constraint: Optional[Interval] = None,
        mixture_scales_prior: Optional[Prior] = None,      
        mixture_scales_constraint: Optional[Interval] = None, 
        mixture_means_prior: Optional[Prior] = None,       
        mixture_means_constraint: Optional[Interval] = None,  
        active_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            batch_shape=batch_shape,
            active_dims=active_dims,
            ard_num_dims=num_dims,
            **kwargs
        )

        for alpha in alphas:
            if not (0 < alpha <= 2):
                raise ValueError(f"所有 alpha 值必须在 (0, 2] 范围内，但收到了 {alpha}")

        self.num_mixtures = len(alphas)
        # 将固定的 alphas 注册为 buffer (不可学习)
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.get_default_dtype()))

        param_shape = list(batch_shape) + [self.num_mixtures, 1, num_dims]

        # --- 可学习参数 ---

        # 混合权重 (w_i)
        if mixture_weights_constraint is None:
            mixture_weights_constraint = Positive() 
        self.register_parameter(
            name="raw_mixture_weights",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, self.num_mixtures))
        )
        self.register_constraint("raw_mixture_weights", mixture_weights_constraint)
        if mixture_weights_prior is not None:
             self.register_prior(
                "mixture_weights_prior", mixture_weights_prior, self._get_mixture_weights, self._set_mixture_weights_unnormalized
             )

        # 混合尺度 (δ_i - delta)
        if mixture_scales_constraint is None:
            mixture_scales_constraint = Positive() 
        self.register_parameter(
            name="raw_mixture_scales",
            parameter=torch.nn.Parameter(torch.zeros(param_shape))
        )
        self.register_constraint("raw_mixture_scales", mixture_scales_constraint)
        if mixture_scales_prior is not None:
             self.register_prior(
                "mixture_scales_prior", mixture_scales_prior, lambda m: m.mixture_scales, lambda m, v: m._set_mixture_scales(v)
             )

        # Spectral Mixture 中对应频率均值 mu
        self.register_parameter(
            name="raw_mixture_means",
            parameter=torch.nn.Parameter(torch.zeros(param_shape))
        )
        if mixture_means_constraint is not None:
            self.register_constraint("raw_mixture_means", mixture_means_constraint)
        if mixture_means_prior is not None:
             self.register_prior(
                "mixture_means_prior", mixture_means_prior, lambda m: m.mixture_means, lambda m, v: m._set_mixture_means(v)
             )


    @property
    def mixture_weights(self):
        raw_val = self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)
        return raw_val / (raw_val.sum(dim=-1, keepdim=True) + 1e-8)

    @mixture_weights.setter
    def mixture_weights(self, value):
        self._set_mixture_weights_unnormalized(value)

    def _get_mixture_weights(self, m):
        return m.mixture_weights

    def _set_mixture_weights_unnormalized(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(value))

    @property
    def mixture_scales(self): 
        return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)

    @mixture_scales.setter
    def mixture_scales(self, value):
        self._set_mixture_scales(value)

    def _set_mixture_scales(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_scales)
        self.initialize(raw_mixture_scales=self.raw_mixture_scales_constraint.inverse_transform(value))

    @property
    def mixture_means(self): # gamma
        constraint = getattr(self, "raw_mixture_means_constraint", None)
        if constraint is not None:
             return constraint.transform(self.raw_mixture_means)
        return self.raw_mixture_means

    @mixture_means.setter
    def mixture_means(self, value):
        self._set_mixture_means(value)

    def _set_mixture_means(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_means)
        constraint = getattr(self, "raw_mixture_means_constraint", None)
        if constraint is not None:
             self.initialize(raw_mixture_means=constraint.inverse_transform(value))
        else:
             self.initialize(raw_mixture_means=value)


    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """
        Initialize mixture components based on batch statistics of the data.

        Args:
            train_x: Training inputs (potentially transformed by model). Shape: batch_shape x n x d
            train_y: Training outputs (potentially transformed by model). Shape: batch_shape x n x m
        """
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")

        # Ensure input tensors have at least 3 dimensions (batch, n, d/m)
        if train_x.dim() < 3:
             train_x = train_x.unsqueeze(0)
        if train_y.dim() < 3:
             if train_y.dim() == 1: 
                 train_y = train_y.unsqueeze(0).unsqueeze(-1)
             elif train_y.dim() == 2: 
                 train_y = train_y.unsqueeze(0)
        if train_y.shape[0] != train_x.shape[0]:
             train_y = train_y.expand(train_x.shape[0], *train_y.shape[1:])


        with torch.no_grad():
            if self.active_dims is not None:
                train_x = train_x[..., self.active_dims]

            train_x = train_x.detach()
            train_y = train_y.detach()

            train_x_sort = train_x.sort(dim=-2)[0]
            max_dist = train_x_sort[..., -1, :] - train_x_sort[..., 0, :] 
            dists = train_x_sort[..., 1:, :] - train_x_sort[..., :-1, :]
            dists = torch.where(dists.eq(0.0), torch.full_like(dists, 1e10), dists)
            min_dist = dists.min(dim=-2)[0] # shape: batch_shape x d

            target_param_shape = self.raw_mixture_scales.shape 
            target_weights_shape = self.raw_mixture_weights.shape 
            num_param_batch_dims = self.raw_mixture_scales.dim() - 3 

            # Reshape min/max dist to match parameter dimensions
            min_dist_prep = min_dist.unsqueeze(-2) # batch_shape x 1 x d
            max_dist_prep = max_dist.unsqueeze(-2) # batch_shape x 1 x d
            min_dist_prep = min_dist_prep.unsqueeze(-3) # batch_shape x 1 x 1 x d
            max_dist_prep = max_dist_prep.unsqueeze(-3) # batch_shape x 1 x 1 x d

            # 3. Squeeze extra batch dims if parameters don't have them
            current_batch_dims = min_dist_prep.dim() - 3
            if num_param_batch_dims < current_batch_dims:
                num_squeeze = current_batch_dims - num_param_batch_dims
                for _ in range(num_squeeze):
                    min_dist_prep = min_dist_prep.squeeze(0)
                    max_dist_prep = max_dist_prep.squeeze(0)
            # Now min/max_dist_prep shape should match target batch_shape + [1, 1, d]

            # 4. Expand to the full target shape
            min_dist_expanded = min_dist_prep.expand(target_param_shape)
            max_dist_expanded = max_dist_prep.expand(target_param_shape)


            # --- Initialize parameters ---
            weights_param_batch_dims = self.raw_mixture_weights.dim() - 1
            train_y_std = train_y.std(dim=(-2, -1)) # batch_shape

            # Align batch dimensions for weights
            current_y_batch_dims = train_y_std.dim()
            if weights_param_batch_dims < current_y_batch_dims:
                 num_squeeze = current_y_batch_dims - weights_param_batch_dims
                 for _ in range(num_squeeze):
                     train_y_std = train_y_std.squeeze(0) 
            target_weights = (train_y_std / self.num_mixtures).unsqueeze(-1).expand(target_weights_shape)
            self.mixture_weights = target_weights

            # Scales initialization
            rand_factor = torch.randn_like(self.raw_mixture_scales) * 0.1 + 1.0
            target_scales = (max_dist_expanded.clamp(min=1e-6).reciprocal_() * rand_factor).abs()
            self.mixture_scales = target_scales

            # Means initialization
            target_means = torch.randn_like(self.raw_mixture_means) * 0.5 / min_dist_expanded.clamp(min=1e-6)
            constraint = getattr(self, "raw_mixture_means_constraint", None)
            if constraint is not None and isinstance(constraint, Positive):
                 target_means = target_means.abs() + 1e-6
            self.mixture_means = target_means

    # forward method remains exactly the same as before
    def forward(self, x1, x2, diag=False, **params):
        if x1.ndim == 1: x1 = x1.unsqueeze(-1)
        if x2.ndim == 1: x2 = x2.unsqueeze(-1)
        if self.ard_num_dims is not None:
             if x1.shape[-1] != self.ard_num_dims:
                 raise RuntimeError(f"x1 需要 {self.ard_num_dims} 维度, 得到 {x1.shape[-1]}")
             if x2.shape[-1] != self.ard_num_dims:
                 raise RuntimeError(f"x2 需要 {self.ard_num_dims} 维度, 得到 {x2.shape[-1]}")


        # --- 计算 τ 和 diff_d ---
        if diag:
            tau_shape = list(x1.shape[:-1])
            tau = torch.zeros(tau_shape, dtype=x1.dtype, device=x1.device)
            diff_d = torch.zeros_like(x1) 
        else:
            # 计算所有点对的差值: batch_shape x n x m x d
            diff_d = x1.unsqueeze(-2) - x2.unsqueeze(-3)
            tau = diff_d.norm(dim=-1) 

        weights = self.mixture_weights
        deltas = self.mixture_scales
        gammas = self.mixture_means
        alphas = self.alphas

        output_shape = list(x1.shape[:-2]) + list(tau.shape[-2:]) 
        kernel_res = torch.zeros(output_shape, dtype=x1.dtype, device=x1.device)

        for i in range(self.num_mixtures):
            alpha_i = alphas[i]
            weight_view_shape = list(weights.shape[:-1]) + [1] * tau.dim() 
            w_i = weights[..., i].view(weight_view_shape)

            delta_i = deltas[..., i, :, :] 
            gamma_i = gammas[..., i, :, :] 

            delta_i_broadcast = delta_i
            gamma_i_broadcast = gamma_i
            if not diag:
                delta_i_broadcast = delta_i.unsqueeze(-2) # batch_shape x 1 x 1 x d
                gamma_i_broadcast = gamma_i.unsqueeze(-2) # batch_shape x 1 x 1 x d
            else: 
                delta_i_broadcast = delta_i # Already batch_shape x 1 x d
                gamma_i_broadcast = gamma_i # Already batch_shape x 1 x d


            abs_diff_d = diff_d.abs()
            epsilon = 1e-8
            term1_input = 2 * math.pi * delta_i_broadcast * abs_diff_d # batch x n x [m] x d
            alpha_i_tensor = torch.tensor(alpha_i, device=term1_input.device, dtype=term1_input.dtype)

            if alpha_i < 1.0:
                term1_input = torch.where(term1_input < epsilon, term1_input + epsilon, term1_input)

            term1_base = term1_input.pow(alpha_i_tensor) # Apply power element-wise

            exp_arg = term1_base.sum(dim=-1) 

            # 计算余弦项参数: sum_d (2πγ_id * diff_d_d)
            term2 = 2 * math.pi * gamma_i_broadcast * diff_d # batch x n x [m] x d
            cos_arg = term2.sum(dim=-1) # Sum over dimensions d -> batch x n x [m]

            kernel_i = w_i * torch.exp(-exp_arg) * torch.cos(cos_arg)
            kernel_res = kernel_res + kernel_i

        if diag:
            pass 

        return kernel_res
    def initialize_from_data_empspect(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """
        Initialize mixture components based on the empirical spectrum of the data.
        Assumes train_x is reasonably gridded for FFT to be meaningful.
        """
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")

        # --- 数据预处理
        if train_x.dim() < 3:
             train_x = train_x.unsqueeze(0)
        if train_y.dim() < 3:
             if train_y.dim() == 1:
                 train_y = train_y.unsqueeze(0).unsqueeze(-1)
             elif train_y.dim() == 2:
                 train_y = train_y.unsqueeze(0)
        if train_y.shape[0] != train_x.shape[0]:
             train_y = train_y.expand(train_x.shape[0], *train_y.shape[1:])

        with torch.no_grad():
            # Detach
            train_x_detached = train_x.detach()
            train_y_detached = train_y.detach()

            # Assuming train_y is (batch_shape x n x m), we take the mean over m if m > 1
            if train_y_detached.shape[-1] > 1:
                y_for_fft = train_y_detached.mean(dim=-1) # Shape: batch_shape x n
            else:
                y_for_fft = train_y_detached.squeeze(-1)   # Shape: batch_shape x n

            # Initialize lists to store parameters for each batch element
            batch_size = y_for_fft.shape[0]
            all_batch_means = []
            all_batch_scales = []
            all_batch_weights = []

            for b in range(batch_size):
                current_y = y_for_fft[b].cpu().numpy() # Process one batch element at a time
                N = len(current_y)
                if N < 2: # FFT needs at least 2 points
                    print(f"Warning: Not enough data points ({N}) in batch {b} for FFT-based initialization. Skipping.")
                    all_batch_means.append(self.mixture_means[b].clone())
                    all_batch_scales.append(self.mixture_scales[b].clone())
                    all_batch_weights.append(self.mixture_weights[b].clone())
                    continue

                emp_spect = np.abs(fft(current_y))**2 / N
                dt = 1.0 # 默认采样间隔
                # train_x_detached shape: batch_shape x n x d
                if train_x_detached.shape[-1] > 0: # 检查是否有输入维度 (d > 0)
                    x_coords_for_dt_estimation = train_x_detached[b, :, 0].cpu().numpy()

                    if len(x_coords_for_dt_estimation) > 1:
                        # 使用最大最小范围来估计平均间隔
                        x_min = x_coords_for_dt_estimation.min()
                        x_max = x_coords_for_dt_estimation.max()
                        if (x_max - x_min) > 1e-8: 
                            dt_est = (x_max - x_min) / (len(x_coords_for_dt_estimation) - 1 + 1e-8)
                            if dt_est > 1e-8:
                                dt = dt_est
                            else:
                                print(f"Warning: Estimated dt for batch {b} based on dim 0 is too small ({dt_est:.2e}). Defaulting dt to 1.0.")
                                dt = 1.0
                        else:
                            print(f"Warning: Range of x_coords for batch {b}, dim 0 is too small to estimate dt. Defaulting dt to 1.0.")
                            dt = 1.0
                    else: # len(x_coords_for_dt_estimation) <= 1
                         print(f"Warning: Not enough x_coords in batch {b}, dim 0 to estimate dt. Defaulting dt to 1.0.")
                         dt = 1.0
                else: 
                    print("Warning: train_x has no spatial/feature dimensions to estimate dt. Defaulting dt to 1.0.")
                    dt = 1.0
                
                freq = np.fft.rfftfreq(N, d=1.0) 
                emp_spect = emp_spect[:len(freq)]

                if len(emp_spect) < self.num_mixtures:
                    print(f"Warning: Not enough frequency components ({len(emp_spect)}) "
                          f"to initialize {self.num_mixtures} mixtures for batch {b}. Using available components.")
                    num_peaks_to_find = len(emp_spect)
                else:
                    num_peaks_to_find = self.num_mixtures

                if num_peaks_to_find == 0: # Should not happen if N >= 2
                    all_batch_means.append(self.mixture_means[b].clone())
                    all_batch_scales.append(self.mixture_scales[b].clone())
                    all_batch_weights.append(self.mixture_weights[b].clone())
                    continue


                # Find peaks in the empirical spectrum
                peak_indices = np.argsort(emp_spect)[-num_peaks_to_find:]
                
                # Initial means (gamma) from peak frequencies
                init_means = freq[peak_indices]

                # Initial scales (delta) - heuristic based on energy/width around peaks
                # A simple heuristic: inverse of frequency (like a period) or related to peak width
                # Let's use a simpler approach similar to CSM: sqrt(energy)/max_energy as a proxy for scale
                # Or, for a Cauchy-like spectrum, scale is related to half-width at half-maximum.
                # For stable kernels, delta is more like 1/lengthscale_in_spectral_domain.
                # Larger delta means narrower spectral peak.
                # We can try: scales ~ 1 / (freq_spacing * peak_width_estimate)
                # Or simpler: scales inversely proportional to the means (higher freq -> narrower peak in s)
                # To avoid division by zero for mean=0, add a small epsilon.
                # Let's use the heuristic from original CSM, it's a starting point.
                init_scales = np.sqrt(emp_spect[peak_indices] / (np.max(emp_spect) + 1e-8)) * 0.5 
                init_scales = np.abs(init_scales) + 1e-6


                # Initial weights from peak energies (normalized)
                init_weights = emp_spect[peak_indices]
                if np.sum(init_weights) > 1e-8:
                    init_weights = init_weights / np.sum(init_weights)
                else: # Avoid division by zero if all emp_spect at peaks are tiny
                    init_weights = np.ones(num_peaks_to_find) / num_peaks_to_find

                # If we found fewer peaks than num_mixtures, pad with reasonable defaults
                if num_peaks_to_find < self.num_mixtures:
                    num_pad = self.num_mixtures - num_peaks_to_find
                    # Pad means with small random positive values or zeros
                    init_means = np.concatenate([init_means, np.random.rand(num_pad) * 0.1])
                    # Pad scales with moderate values
                    init_scales = np.concatenate([init_scales, np.ones(num_pad) * 0.5])
                    # Pad weights (distribute remaining weight or use small equal weights)
                    init_weights = np.concatenate([init_weights, np.ones(num_pad) / self.num_mixtures])
                    init_weights /= np.sum(init_weights) # Re-normalize


                # Convert to tensors and reshape for ARD (repeating for each dimension)
                # Target shape for means/scales: num_mixtures x 1 x ard_num_dims
                # Target shape for weights: num_mixtures
                dtype = self.raw_mixture_means.dtype
                device = self.raw_mixture_means.device

                # For ARD, replicate the 1D initialized params across all dims
                # Means: num_mixtures -> num_mixtures x 1 x ard_num_dims
                batch_init_means = torch.tensor(init_means, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
                batch_init_means = batch_init_means.expand(self.num_mixtures, 1, self.ard_num_dims)
                all_batch_means.append(batch_init_means)

                # Scales: num_mixtures -> num_mixtures x 1 x ard_num_dims
                batch_init_scales = torch.tensor(init_scales, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
                batch_init_scales = batch_init_scales.expand(self.num_mixtures, 1, self.ard_num_dims)
                all_batch_scales.append(batch_init_scales)

                # Weights: num_mixtures
                batch_init_weights = torch.tensor(init_weights, dtype=dtype, device=device)
                all_batch_weights.append(batch_init_weights)

            if batch_size > 0:
                # Stack along the batch dimension
                final_means = torch.stack(all_batch_means, dim=0)
                final_scales = torch.stack(all_batch_scales, dim=0)
                final_weights = torch.stack(all_batch_weights, dim=0)

                # Use setters to apply constraints and initialize raw parameters
                self.mixture_means = final_means
                self.mixture_scales = final_scales
                self.mixture_weights = final_weights # Setter will handle unnormalization if needed
                print("MixedFixedAlphaStableKernel initialized from empirical spectrum.")
            else:
                print("Warning: No batches to initialize from empirical spectrum.")
class LearnableAlphaStableKernel(Kernel):
    r"""
    一个具有可学习的稳定性参数 alpha 的稳定谱混合核。
    该核有 Q 个混合组件，每个组件的 alpha, weight, mean, scale 都是可学习的。
    (版本：在解决 'gamma全为零' 问题之前)
    """
    is_stationary = True
    has_lengthscale = False

    def __init__(
        self,
        num_mixtures: int, # Q, 组件数
        num_dims: int,     # d, 输入维度
        batch_shape: Optional[torch.Size] = torch.Size([]),
        alphas_prior: Optional[Prior] = None,
        alphas_constraint: Optional[Interval] = None,
        mixture_weights_prior: Optional[Prior] = None,
        mixture_weights_constraint: Optional[Interval] = None,
        mixture_scales_prior: Optional[Prior] = None,
        mixture_scales_constraint: Optional[Interval] = None,
        mixture_means_prior: Optional[Prior] = None,
        mixture_means_constraint: Optional[Interval] = None,
        active_dims: Optional[List[int]] = None,
        initial_dt_override: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            batch_shape=batch_shape,
            active_dims=active_dims,
            ard_num_dims=num_dims,
            **kwargs
        )
        self.num_mixtures = num_mixtures
        self.fixed_dt_for_run = initial_dt_override
        
        param_shape_ard = list(batch_shape) + [self.num_mixtures, 1, num_dims]
        param_shape_scalar = list(batch_shape) + [self.num_mixtures]

        # --- 参数注册和初始化流程 (已修复 AttributeError 和 ValueError) ---
        # 1. Alphas
        self.register_parameter(name="raw_alphas", parameter=torch.nn.Parameter(torch.zeros(param_shape_scalar)))
        if alphas_constraint is None: alphas_constraint = Interval(0.01, 2.0)
        self.register_constraint("raw_alphas", alphas_constraint)
        if alphas_prior is not None: self.register_prior("alphas_prior", alphas_prior, lambda m: m.alphas, lambda m, v: m._set_alphas(v))
        if self.num_mixtures > 1:
            initial_alphas_1d = torch.linspace(1.0, 2.0, self.num_mixtures)
        else:
            initial_alphas_1d = torch.tensor([1.5])
        view_shape = [1] * len(batch_shape) + [self.num_mixtures]
        initial_alphas = initial_alphas_1d.view(*view_shape).expand(*param_shape_scalar)
        print(f"Initializing learnable alphas with diverse values: {initial_alphas_1d.numpy()}")
        self.alphas = initial_alphas

        # 2. Weights
        self.register_parameter(name="raw_mixture_weights", parameter=torch.nn.Parameter(torch.zeros(param_shape_scalar)))
        if mixture_weights_constraint is None: mixture_weights_constraint = Positive()
        self.register_constraint("raw_mixture_weights", mixture_weights_constraint)
        if mixture_weights_prior is not None: self.register_prior("mixture_weights_prior", mixture_weights_prior, self._get_mixture_weights, self._set_mixture_weights_unnormalized)
        self.mixture_weights = torch.ones(param_shape_scalar)

        # 3. Scales
        self.register_parameter(name="raw_mixture_scales", parameter=torch.nn.Parameter(torch.zeros(param_shape_ard)))
        if mixture_scales_constraint is None: mixture_scales_constraint = GreaterThan(1e-6)
        self.register_constraint("raw_mixture_scales", mixture_scales_constraint)
        if mixture_scales_prior is None: mixture_scales_prior = LogNormalPrior(0.0, 1.0)
        if mixture_scales_prior is not None: self.register_prior("mixture_scales_prior", mixture_scales_prior, lambda m: m.mixture_scales, lambda m, v: m._set_mixture_scales(v))
        self.mixture_scales = torch.full(param_shape_ard, 0.1)

        # 4. Means
        self.register_parameter(name="raw_mixture_means", parameter=torch.nn.Parameter(torch.zeros(param_shape_ard)))
        if mixture_means_constraint is not None: self.register_constraint("raw_mixture_means", mixture_means_constraint)
        if mixture_means_prior is None: mixture_means_prior = NormalPrior(0.0, 1.0)
        if mixture_means_prior is not None: self.register_prior("mixture_means_prior", mixture_means_prior, lambda m: m.mixture_means, lambda m, v: m._set_mixture_means(v))
        self.mixture_means = torch.randn(param_shape_ard) * 0.01

    # --- property, setter, 和 forward 方法 (保持不变) ---
    @property
    def alphas(self): return self.raw_alphas_constraint.transform(self.raw_alphas)
    @alphas.setter
    def alphas(self, value):
        if not torch.is_tensor(value): value = torch.as_tensor(value).to(self.raw_alphas)
        self.initialize(raw_alphas=self.raw_alphas_constraint.inverse_transform(value))
    def _set_alphas(self, value): self.alphas = value

    @property
    def mixture_weights(self):
        raw_val = self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)
        return raw_val / (raw_val.sum(dim=-1, keepdim=True) + 1e-8)
    @mixture_weights.setter
    def mixture_weights(self, value):
        if not torch.is_tensor(value): value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(value))
    def _get_mixture_weights(self, m): return m.mixture_weights
    def _set_mixture_weights_unnormalized(self, value): self.mixture_weights = value
    
    @property
    def mixture_scales(self): return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)
    @mixture_scales.setter
    def mixture_scales(self, value):
        if not torch.is_tensor(value): value = torch.as_tensor(value).to(self.raw_mixture_scales)
        self.initialize(raw_mixture_scales=self.raw_mixture_scales_constraint.inverse_transform(value))

    @property
    def mixture_means(self):
        constraint = getattr(self, "raw_mixture_means_constraint", None)
        return constraint.transform(self.raw_mixture_means) if constraint is not None else self.raw_mixture_means
    @mixture_means.setter
    def mixture_means(self, value):
        if not torch.is_tensor(value): value = torch.as_tensor(value).to(self.raw_mixture_means)
        constraint = getattr(self, "raw_mixture_means_constraint", None)
        if constraint is not None: self.initialize(raw_mixture_means=constraint.inverse_transform(value))
        else: self.initialize(raw_mixture_means=value)
    def forward(self, x1, x2, diag=False, **params):
        if x1.ndim == 1: x1 = x1.unsqueeze(-1)
        if x2.ndim == 1: x2 = x2.unsqueeze(-1)
        if self.ard_num_dims is not None:
             if x1.shape[-1] != self.ard_num_dims: raise RuntimeError(f"x1 需要 {self.ard_num_dims} 维度, 得到 {x1.shape[-1]}")
             if x2.shape[-1] != self.ard_num_dims: raise RuntimeError(f"x2 需要 {self.ard_num_dims} 维度, 得到 {x2.shape[-1]}")
        if diag:
            tau = torch.zeros(list(x1.shape[:-1]), dtype=x1.dtype, device=x1.device)
            diff_d = torch.zeros_like(x1)
        else:
            diff_d = x1.unsqueeze(-2) - x2.unsqueeze(-3)
            tau = diff_d.norm(dim=-1)
        weights = self.mixture_weights
        deltas = self.mixture_scales
        gammas = self.mixture_means
        alphas = self.alphas
        output_shape = list(x1.shape[:-2]) + list(tau.shape[-2:])
        kernel_res = torch.zeros(output_shape, dtype=x1.dtype, device=x1.device)
        for i in range(self.num_mixtures):
            alpha_i = alphas[..., i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weight_view_shape = list(weights.shape[:-1]) + [1] * tau.dim()
            w_i = weights[..., i].view(weight_view_shape)
            delta_i = deltas[..., i, :, :]
            gamma_i = gammas[..., i, :, :]
            delta_i_broadcast = delta_i.unsqueeze(-2) if not diag else delta_i
            gamma_i_broadcast = gamma_i.unsqueeze(-2) if not diag else gamma_i
            abs_diff_d = diff_d.abs()
            epsilon = 1e-8
            term1_input = 2 * math.pi * delta_i_broadcast * abs_diff_d
            if torch.any(alphas[..., i] < 1.0): # 更安全的检查
                term1_input = torch.where(term1_input < epsilon, term1_input + epsilon, term1_input)
            term1_base = term1_input.pow(alpha_i.squeeze())
            exp_arg = term1_base.sum(dim=-1)
            term2 = 2 * math.pi * gamma_i_broadcast * diff_d
            cos_arg = term2.sum(dim=-1)
            kernel_i = w_i * torch.exp(-exp_arg) * torch.cos(cos_arg)
            kernel_res = kernel_res + kernel_i
        return kernel_res
    
    def initialize_from_data_empspect(self, train_x: torch.Tensor, train_y: torch.Tensor):
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")

        if train_x.dim() < 3: train_x = train_x.unsqueeze(0)
        if train_y.dim() < 3:
             if train_y.dim() == 1: train_y = train_y.unsqueeze(0).unsqueeze(-1)
             elif train_y.dim() == 2: train_y = train_y.unsqueeze(0)
        if train_y.shape[0] != train_x.shape[0]:
             train_y = train_y.expand(train_x.shape[0], *train_y.shape[1:])

        with torch.no_grad():
            train_x_detached = train_x.detach()
            train_y_detached = train_y.detach()

            if train_y_detached.shape[-1] > 1:
                y_for_fft = train_y_detached.mean(dim=-1)
            else:
                y_for_fft = train_y_detached.squeeze(-1)

            batch_size = y_for_fft.shape[0]
            all_batch_means, all_batch_scales, all_batch_weights = [], [], []
            
            for b in range(batch_size):
                current_y = y_for_fft[b].cpu().numpy()
                N = len(current_y)
                
                def fallback_init():
                    print(f"  Fallback: Using random initialization for batch {b}.")
                    rand_means = torch.rand(self.num_mixtures, 1, self.ard_num_dims, device=self.raw_mixture_means.device, dtype=self.raw_mixture_means.dtype) * 0.5
                    rand_scales = torch.rand(self.num_mixtures, 1, self.ard_num_dims, device=self.raw_mixture_scales.device, dtype=self.raw_mixture_scales.dtype) * 0.5 + 0.1
                    rand_weights = torch.ones(self.num_mixtures, device=self.raw_mixture_weights.device, dtype=self.raw_mixture_weights.dtype) / self.num_mixtures
                    all_batch_means.append(rand_means); all_batch_scales.append(rand_scales); all_batch_weights.append(rand_weights)

                if N < 2:
                    print(f"Warning: Not enough data points ({N}) in batch {b} for FFT.")
                    fallback_init(); continue

                emp_spect_full = np.abs(fft(current_y))**2 / N
                dt = self.fixed_dt_for_run if self.fixed_dt_for_run is not None else 1.0
                freq = np.fft.rfftfreq(N, d=dt)
                emp_spect = emp_spect_full[:len(freq)]

                if len(emp_spect) < 1:
                    print(f"Warning: Zero frequency components after rfftfreq for batch {b}.")
                    fallback_init(); continue

                num_peaks_to_find = self.num_mixtures
                if len(emp_spect) < self.num_mixtures:
                    print(f"Warning: Not enough frequency components ({len(emp_spect)}) to initialize {self.num_mixtures} mixtures. Using all {len(emp_spect)} components.")
                    num_peaks_to_find = len(emp_spect)
                
                if num_peaks_to_find == 0:
                    print(f"Warning: num_peaks_to_find is 0 for batch {b}. Using fallback init.")
                    fallback_init(); continue

                peak_indices = np.argsort(emp_spect)[-num_peaks_to_find:]
                init_means = freq[peak_indices]
                max_emp_spect = np.max(emp_spect); max_emp_spect = max_emp_spect if max_emp_spect > 1e-8 else 1e-8
                init_scales = np.sqrt(emp_spect[peak_indices] / max_emp_spect) * 0.5; init_scales = np.abs(init_scales) + 1e-6
                init_weights = emp_spect[peak_indices]
                sum_init_weights = np.sum(init_weights)
                if sum_init_weights > 1e-8: init_weights /= sum_init_weights
                else: init_weights = np.ones(num_peaks_found) / num_peaks_found if num_peaks_found > 0 else np.array([]) 

                if len(init_means) < self.num_mixtures:
                    num_pad = self.num_mixtures - len(init_means)
                    print(f"  Padding with {num_pad} default component(s).")
                    
                    # 填充均值：在已找到均值的范围内随机取，或者一个小的随机值
                    if len(init_means) > 0:
                        mean_range = (np.min(init_means), np.max(init_means))
                    else: # 如果一个峰值都没找到
                        mean_range = (0.01, 0.1)
                    pad_means = np.random.uniform(mean_range[0], mean_range[1] + 0.1, num_pad)
                    init_means = np.concatenate([init_means, pad_means])
                    
                    # 填充尺度：使用已找到尺度的中位数，或者一个合理的默认值
                    if len(init_scales) > 0:
                        median_scale = np.median(init_scales)
                    else:
                        median_scale = 0.5
                    pad_scales = np.full(num_pad, median_scale)
                    init_scales = np.concatenate([init_scales, pad_scales])
                    
                    # 填充权重：赋予一个小的基础权重，然后重新归一化
                    if len(init_weights) > 0:
                        # 假设填充的组件重要性较低，给一个较小的值
                        pad_weights = np.full(num_pad, np.mean(init_weights) * 0.1 + 1e-6)
                    else:
                        pad_weights = np.full(num_pad, 1.0 / self.num_mixtures)
                    init_weights = np.concatenate([init_weights, pad_weights])
                    init_weights /= np.sum(init_weights) # 重新归一化所有权重
                if len(init_means) != self.num_mixtures:
                    print(f"Error: After padding, parameter lengths ({len(init_means)}) do not match num_mixtures ({self.num_mixtures}).")
                    fallback_init() 
                    continue      

                dtype = self.raw_mixture_means.dtype; device = self.raw_mixture_means.device
                batch_init_means = torch.tensor(init_means, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1).expand(self.num_mixtures, 1, self.ard_num_dims)
                all_batch_means.append(batch_init_means)
                batch_init_scales = torch.tensor(init_scales, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1).expand(self.num_mixtures, 1, self.ard_num_dims)
                all_batch_scales.append(batch_init_scales)
                batch_init_weights = torch.tensor(init_weights, dtype=dtype, device=device)
                all_batch_weights.append(batch_init_weights)

            if all_batch_means and len(all_batch_means) == batch_size:
                self.mixture_means = torch.stack(all_batch_means, dim=0)
                self.mixture_scales = torch.stack(all_batch_scales, dim=0)
                self.mixture_weights = torch.stack(all_batch_weights, dim=0)
                print(f"{self.__class__.__name__} parameters initialized from empirical spectrum.")
            else:
                print(f"Warning: Empirical spectrum initialization failed for {self.__class__.__name__}. Using default/random init.")

