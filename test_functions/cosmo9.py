import torch
from classy import Class
from .optimization_problem import OptimizationProblem


class Cosmo9(OptimizationProblem):
    """Cosmological Constraints function implemented in PyTorch using CLASS

    This function evaluates the negative log-likelihood based on the cosmological
    parameters provided and compares the theoretical results (from CLASS) to
    observational data.

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value (not known for this problem)
    :ivar info: String with problem info
    """

    def __init__(self):
        self.dim = 9
        self.lb = torch.tensor([0.02, 0.1, 0.5, 0.92, 1e-9, 0.04, -0.01, -1.5, -0.5])  # Lower bounds
        self.ub = torch.tensor([0.024, 0.14, 0.9, 1.0, 3e-9, 0.08, 0.01, -0.5, 0.5])   # Upper bounds
        self.int_var = torch.tensor([], dtype=torch.int32)
        self.cont_var = torch.arange(0, 9)
        self.info = (
            "9-dimensional Cosmological Constraints function using CLASS\n"
            + "Evaluate negative log-likelihood for SDSS constraints"
        )
        self.min = None  # Global minimum is not defined for this problem

    def eval(self, x):
        """Evaluate the Cosmological Constraints function at x

        :param x: Data points
        :type x: torch.Tensor of shape (n, 9)
        :return: Negative log-likelihood values at x
        :rtype: torch.Tensor of shape (n, 1)
        """
        self.__check_input__(x)
        n = x.shape[0]
        results = torch.zeros(n, 1)

        # Evaluate each input point
        for i in range(n):
            params = x[i].tolist()
            results[i] = self.__evaluate_single_point__(params)

        return results

    def __evaluate_single_point__(self, params):
        """Evaluate the objective for a single parameter set"""
        cosmo = Class()
        # Convert parameters to CLASS format
        settings = {
            'output': 'tCl,lCl,mPk',
            'l_max_scalars': 2000,
            'omega_b': params[0],   # Omega_b
            'omega_cdm': params[1], # Omega_cdm
            'h': params[2],         # Hubble parameter
            'n_s': params[3],       # Scalar spectral index
            'A_s': params[4],       # Scalar amplitude
            'tau_reio': params[5],  # Reionization optical depth
            'Omega_k': params[6],   # Curvature parameter
            'w0_fld': params[7],    # Dark energy equation of state parameter w0
            'wa_fld': params[8]     # Dark energy equation of state parameter wa
        }

        try:
            # Initialize CLASS and compute theoretical values
            cosmo.set(settings)
            cosmo.compute()
            cl = cosmo.lensed_cl()  # CMB power spectrum
            cl_tt = cl['tt'][2:]    # Extract TT power spectrum

            # Load observational data
            obs_data = torch.load("data/observed_cl.pt")  # Observed CMB TT data
            cov_matrix = torch.load("data/covariance_matrix.pt")  # Covariance matrix
            residual = torch.tensor(cl_tt) - obs_data

            # Compute negative log-likelihood
            log_likelihood = -0.5 * torch.matmul(
                torch.matmul(residual.T, torch.linalg.inv(cov_matrix)), residual
            )

            return -log_likelihood.item()

        except Exception as e:
            print(f"Error in CLASS computation: {e}")
            return float('inf')

        finally:
            cosmo.struct_cleanup()
            cosmo.empty()

    def __check_input__(self, x):
        """Internal function to check input validity."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.shape[1] != self.dim:
            raise ValueError(f"Input must have shape (n, {self.dim})")
        if torch.any(x < self.lb) or torch.any(x > self.ub):
            raise ValueError("Input values must be within the bounds")
