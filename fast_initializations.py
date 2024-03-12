# Imports
import torch
import numpy as np
import torch.nn.functional as F
from torch.linalg import cholesky

import kernels
import fast_recognition
from flexible_multivariate_normal import vector_to_tril, vector_to_tril_diag_idx
from utils import diagonalize
from flexible_multivariate_normal import tril_to_vector
from prior import GPPrior


def _default_field(param: dict, key: str, default):
    if key not in param:
        param[key] = default


def _repeat_list(val, num):
    return [val for _ in range(num)]


class Mixin:
    """
        Mixin class containing necessary methods for initializing RPGPFA model
    """

    def _init_fit_params(self):
        """ Default Fit parameters """

        # Init dictionary
        if self.fit_params is None:
            self.fit_params = {}

        # Number of conditionally independent factors
        num_factors = self.num_factors

        # Default optimizer / scheduler
        optimizer_closure_default = lambda params: torch.optim.Adam(params, lr=1e-3)
        scheduler_closure_default = lambda optim: torch.optim.lr_scheduler.ConstantLR(optim, factor=1)

        # Latent dimensions
        _default_field(self.fit_params, key='dim_latent', default=1)

        # Iterations
        _default_field(self.fit_params, key='num_epoch', default=500)

        # Batch size (default to full batch)
        _default_field(self.fit_params, key='batch_size', default=self.num_observation)

        # Logger
        _default_field(self.fit_params, key='pct', default=0.01)

        # Ergodic assumption
        _default_field(self.fit_params, key='ergodic', default=False)

        # Default Prior Parameters
        _default_field(self.fit_params, key ='prior_params', default ={})
        # Kernel Types
        _default_field(self.fit_params['prior_params'], key='gp_kernel', default='RBF')
        # Fit/Fix Parameters
        _default_field(self.fit_params['prior_params'], key='fit_kernel_scale', default=False)
        _default_field(self.fit_params['prior_params'], key='fit_kernel_scale_prior', default=False)
        _default_field(self.fit_params['prior_params'], key='fit_kernel_lengthscale', default=True)
        _default_field(self.fit_params['prior_params'], key='fit_kernel_lengthscale_prior', default=True)
        _default_field(self.fit_params['prior_params'], key='fit_prior_mean_param', default=True)
        # Set Default/Init Parameters
        _default_field(self.fit_params['prior_params'], key='scale', default=1.0)
        _default_field(self.fit_params['prior_params'], key='scale_prior', default=1.0)
        _default_field(self.fit_params['prior_params'], key='lengthscale', default=0.01)
        _default_field(self.fit_params['prior_params'], key='lengthscale_prior', default=0.02)
        # Optimizer
        _default_field(self.fit_params['prior_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['prior_params'], key='scheduler', default=scheduler_closure_default)

        # Default Recognition Factors Parameters
        _default_field(self.fit_params, key='factors_params', default={})
        # Network Parameters
        _default_field(self.fit_params['factors_params'], key='channels', default= _repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='kernel_conv', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='kernel_pool', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='dim_hidden', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='nonlinearity', default=_repeat_list(F.relu, num_factors))
        _default_field(self.fit_params['factors_params'], key='covariance', default=_repeat_list('fixed', num_factors))
        # Dropout
        _default_field(self.fit_params['factors_params'], key='dropout', default=0.0)
        # Optimizer
        _default_field(self.fit_params['factors_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['factors_params'], key='scheduler', default=scheduler_closure_default)


        # Default Recognition Auxiliary Factors Parameters
        _default_field(self.fit_params, key='auxiliary_params', default={})
        # Network Parameters
        _default_field(self.fit_params['auxiliary_params'], key='channels', default= _repeat_list((), num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='kernel_conv', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='kernel_pool', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='dim_hidden', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='nonlinearity', default=_repeat_list(F.relu, num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='covariance', default=_repeat_list('fixed', num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='dropout', default=0.0)
        _default_field(self.fit_params['auxiliary_params'], key='zero_init', default=False)
        _default_field(self.fit_params['auxiliary_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['auxiliary_params'], key='scheduler', default=scheduler_closure_default)
        _default_field(self.fit_params['auxiliary_params'], key='fixed', default=False)


        # Default Variational Parameters
        _default_field(self.fit_params, key='variational_params', default={})
        _default_field(self.fit_params['variational_params'], key='inference_mode', default='amortized')
        # Network Parameters
        _default_field(self.fit_params['variational_params'], key='channels', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['variational_params'], key='kernel_conv', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['variational_params'], key='kernel_pool', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['variational_params'], key='dim_hidden', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['variational_params'], key='dim_hidden_merged', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['variational_params'], key='nonlinearity', default=_repeat_list(F.relu, num_factors))
        _default_field(self.fit_params['variational_params'], key='nonlinearity_merged', default=F.relu)
        _default_field(self.fit_params['variational_params'], key='covariance', default=_repeat_list('fixed', num_factors))
        _default_field(self.fit_params['variational_params'], key='dropout', default=0.0)
        _default_field(self.fit_params['variational_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['variational_params'], key='scheduler', default=scheduler_closure_default)

    def _init_prior(self):
        """ Initialise parameters of k=1..K independent kernels """

        natural1 = torch.zeros(self.dim_latent, device=self.device, dtype=self.dtype)
        natural2 = -0.5 * torch.eye(self.dim_latent, device=self.device, dtype=self.dtype)

        self.forwarded_prior = [natural1, natural2]

    def _init_factors(self, observations):
        """ Initialize recognition network of each factor """

        if self.recognition_factors is None:

            dim_inputs = [
                obsi.shape[1:] for obsi in observations
            ]

            # Grasp fit params
            fit_params = self.fit_params['factors_params']
            dim_latent = self.dim_latent
            num_factors = self.num_factors

            # Convolutional parameters
            channels = fit_params["channels"]
            kernel_conv = fit_params["kernel_conv"]
            kernel_pool = fit_params["kernel_pool"]
            dropout = fit_params["dropout"]

            # Fully connected layers parameters
            dim_hidden = fit_params["dim_hidden"]
            non_linearity = fit_params["nonlinearity"]

            # Covariance type
            covariance = fit_params["covariance"]

            # Build and Append networks
            recognition_factors = []
            for obsi in range(num_factors):
                neti = fast_recognition.Net(
                    dim_input=dim_inputs[obsi],
                    dim_latent=dim_latent,
                    kernel_conv=kernel_conv[obsi],
                    kernel_pool=kernel_pool[obsi],
                    channels=channels[obsi],
                    dim_hidden=dim_hidden[obsi],
                    non_linearity=non_linearity[obsi],
                    zero_init=False,
                    dropout=dropout,
                ).to(self.device.index)
                recognition_factors.append(neti)

            self.recognition_factors = recognition_factors

    def _init_auxiliary(self, observations):
        """ Initialize auxiliary recognition network of each factor """

        if self.recognition_auxiliary is None:

            dim_inputs = [
                obsi.shape[1:] for obsi in observations
            ]

            # Grasp fit params
            fit_params = self.fit_params['auxiliary_params']
            dim_latent = self.dim_latent
            num_factors = self.num_factors

            # Convolutional parameters
            channels = fit_params["channels"]
            kernel_conv = fit_params["kernel_conv"]
            kernel_pool = fit_params["kernel_pool"]
            dropout = fit_params["dropout"]

            # Fully connected layers parameters
            dim_hidden = fit_params["dim_hidden"]
            non_linearity = fit_params["nonlinearity"]
            zero_init = fit_params["zero_init"]

            # Covariance type
            covariance = fit_params["covariance"]

            # Build and Append networks
            recognition_auxiliary = []
            for obsi in range(num_factors):
                neti = fast_recognition.Net(
                    dim_input=dim_inputs[obsi],
                    dim_latent=dim_latent,
                    kernel_conv=kernel_conv[obsi],
                    kernel_pool=kernel_pool[obsi],
                    channels=channels[obsi],
                    dim_hidden=dim_hidden[obsi],
                    non_linearity=non_linearity[obsi],
                    zero_init=zero_init,
                    dropout=dropout,
                ).to(self.device.index)
                recognition_auxiliary.append(neti)

            self.recognition_auxiliary = recognition_auxiliary

    def _init_precision_factors(self):
        # Return alpha
        # TODO: add diagonal as an option


        diag_idx = vector_to_tril_diag_idx(self.dim_latent)
        chol = torch.zeros(self.num_factors, int(self.dim_latent * (self.dim_latent + 1) / 2), dtype=self.dtype)
        chol[..., diag_idx] = -np.sqrt(0.5)
        chol_precision = torch.nn.Parameter(vector_to_tril(chol)).to(self.device.index)
        self.precision_chol_factors = chol_precision

    def _init_precision_auxiliary(self):
        diag_idx = vector_to_tril_diag_idx(self.dim_latent)
        chol = torch.zeros(self.num_factors, int(self.dim_latent * (self.dim_latent + 1) / 2), dtype=self.dtype)
        chol[..., diag_idx] = -np.sqrt(0.5)
        chol_precision = torch.nn.Parameter(vector_to_tril(chol)).to(self.device.index)
        self.precision_chol_auxiliary = chol_precision


