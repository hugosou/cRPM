# Imports
import copy
import torch
import numpy as np
import torch.nn.functional as F

from torch import matmul

import fast_recognition
from flexible_multivariate_normal import vector_to_tril, vector_to_tril_diag_idx, tril_to_vector

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

        # Constrain auxiliary
        _default_field(self.fit_params, key='auxiliary_update', default=False)
        _default_field(self.fit_params, key='auxiliary_toggle', default=lambda x: False)
        _default_field(self.fit_params, key='auxiliary_mode', default='flexible')

        # Logger
        _default_field(self.fit_params, key='pct', default=0.01)

        # Default Recognition Factors Parameters
        _default_field(self.fit_params, key='factors_params', default={})
        _default_field(self.fit_params, key='auxiliary_params', default={})

        # Network Parameters
        _default_field(self.fit_params['factors_params'], key='channels', default= _repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='kernel_conv', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='kernel_pool', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='dim_hidden', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='nonlinearity', default=_repeat_list(F.relu, num_factors))
        _default_field(self.fit_params['factors_params'], key='dropout',  default=_repeat_list(0.0, num_factors))

        # Auxiliary Network Parameters
        _default_field(self.fit_params['auxiliary_params'], key='channels', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='kernel_conv', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='kernel_pool', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='dim_hidden', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='nonlinearity', default=_repeat_list(F.relu, num_factors))
        _default_field(self.fit_params['auxiliary_params'], key='dropout', default=_repeat_list(0.0, num_factors))
        
        # Optimizer
        _default_field(self.fit_params['factors_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['factors_params'], key='scheduler', default=scheduler_closure_default)
        _default_field(self.fit_params['auxiliary_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['auxiliary_params'], key='scheduler', default=scheduler_closure_default)

    def _init_recognition(self, fit_params: dict, observations: list):

        dim_inputs = [
            obsi.shape[1:] for obsi in observations
        ]

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

        # Build and Append networks
        rec = []
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
                dropout=dropout[obsi],
            ).to(self.device.index)
            rec.append(neti)

        return rec

    def _init_prior(self):
        """ Initialise parameters of k=1..K independent kernels """

        natural1 = torch.zeros(self.dim_latent, device=self.device, dtype=self.dtype)
        natural2 = -0.5 * torch.eye(self.dim_latent, device=self.device, dtype=self.dtype)

        self.forwarded_prior = [natural1, natural2]

    def _init_factors(self, observations):
        """ Initialize recognition network of each factor """
        if self.recognition_factors is None:
            self.recognition_factors = self._init_recognition(self.fit_params['factors_params'], observations)

    def _init_auxiliary(self, observations):
        """ Initialize auxiliary recognition networks """
        if self.recognition_auxiliary is None:
            self.recognition_auxiliary = copy.deepcopy(self.recognition_factors)
            #self.recognition_auxiliary = self._init_recognition(self.fit_params['auxiliary_params'], observations)

    def _init_precision_factors(self):

        if self.precision_chol_vec_factors is None:
            diag_idx = vector_to_tril_diag_idx(self.dim_latent)
            chol = np.zeros((self.num_factors, int(self.dim_latent * (self.dim_latent + 1) / 2)) )
            chol[:, diag_idx] = np.sqrt(0.5)
            self.precision_chol_vec_factors = fast_recognition.Precision(
                torch.tensor(chol, dtype=self.dtype)
            ).to(self.device.index)

    def _init_precision_auxiliary(self):

        if self.precision_chol_vec_auxiliary is None:

            diag_idx = vector_to_tril_diag_idx(self.dim_latent)
            chol = np.zeros((self.num_factors, int(self.dim_latent * (self.dim_latent + 1) / 2)))
            chol[:, diag_idx] = np.sqrt(0.5)
            self.precision_chol_vec_auxiliary = fast_recognition.Precision(
                torch.tensor(chol, dtype=self.dtype)
            ).to(self.device.index)

            # with torch.no_grad():
            #
            #     # Get factors precision (not yet  offset by prior)
            #     precision_factors_chol = vector_to_tril(self.precision_chol_vec_factors.chol_vec)
            #     precision_factors = - matmul(precision_factors_chol, precision_factors_chol.transpose(-1, -2))
            #
            #     # Precision prior
            #     precision_prior = self.forwarded_prior[1].unsqueeze(0)
            #
            # # Cholesky Decompose
            # delta_chol = torch.linalg.cholesky(
            #     - precision_prior - precision_factors - precision_factors.sum(dim=0, keepdim=True)
            # )
            #
            # # Init
            # self.precision_chol_vec_auxiliary = fast_recognition.Precision(
            #     tril_to_vector(delta_chol)
            # ).to(self.device.index)

            # diag_idx = vector_to_tril_diag_idx(self.dim_latent)
            # chol = np.zeros((self.num_factors, int(self.dim_latent * (self.dim_latent + 1) / 2)))
            # chol[:, diag_idx] = np.sqrt(0.5)
            # # chol[:, diag_idx] = 0
            # chol = 0.01 * np.random.randn(self.num_factors, int(self.dim_latent * (self.dim_latent + 1) / 2))
            # self.precision_chol_vec_auxiliary = torch.tensor(chol, dtype=self.dtype, requires_grad=True, device=self.device)

