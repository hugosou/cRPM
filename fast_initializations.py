# Imports
import copy
import torch
import numpy as np
import torch.nn.functional as F

from torch import matmul

import fast_recognition
from prior import MixturePrior
from flexible_multivariate_normal import vector_to_tril, vector_to_tril_diag_idx, tril_to_vector


def _default_field(param: dict, key: str, default):
    """Set Dictionaries default param if not provided"""
    if key not in param:
        param[key] = default


def _repeat_list(val, num):
    """Repeat element in list"""
    return [val for _ in range(num)]


class Mixin:
    """Mixin class containing necessary methods for initializing fast RPM model"""

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
        _default_field(self.fit_params, key='auxiliary_mode', default='constrained_prior')

        # Logger verbosity (percentage of iteration)
        _default_field(self.fit_params, key='pct', default=0.01)

        # Prior Parameters
        _default_field(self.fit_params, key='prior_params', default={})
        _default_field(self.fit_params['prior_params'], key='num_centroids', default=1)
        _default_field(self.fit_params['prior_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['prior_params'], key='scheduler', default=scheduler_closure_default)

        # Network Parameters
        _default_field(self.fit_params, key='factors_params', default={})
        _default_field(self.fit_params['factors_params'], key='channels', default= _repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='kernel_conv', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='kernel_pool', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='dim_hidden', default=_repeat_list((), num_factors))
        _default_field(self.fit_params['factors_params'], key='non_linearity', default=_repeat_list(F.relu, num_factors))
        _default_field(self.fit_params['factors_params'], key='dropout',  default=_repeat_list(0.0, num_factors))
        _default_field(self.fit_params['factors_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['factors_params'], key='scheduler', default=scheduler_closure_default)

        # Variational Parameters (Necessary for non-closed form updates)
        _default_field(self.fit_params, key='variational_params', default={})
        _default_field(self.fit_params['variational_params'], key='dim_hidden', default=())
        _default_field(self.fit_params['variational_params'], key='non_linearity', default=torch.nn.Identity())
        _default_field(self.fit_params['variational_params'], key='dropout', default=0.0)
        _default_field(self.fit_params['variational_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['variational_params'], key='scheduler', default=scheduler_closure_default)

    def _init_recognition(self, fit_params: dict, observations: list):
        """Recognition Function"""

        # Input dimensions
        dim_inputs = [
            obsi.shape[1:] for obsi in observations
        ]

        # Problem dimensions
        dim_latent = self.dim_latent
        num_factors = self.num_factors

        # Convolutional parameters
        channels = fit_params["channels"]
        kernel_conv = fit_params["kernel_conv"]
        kernel_pool = fit_params["kernel_pool"]
        dropout = fit_params["dropout"]

        # Fully connected layers parameters
        dim_hidden = fit_params["dim_hidden"]
        non_linearity = fit_params["non_linearity"]

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
        """ Prior Distribution: (mixture of) FlexibleMultivariateNormal """

        if self.prior is None:

            # Grasp Params
            params = self.fit_params['prior_params']
            num_centroids = params['num_centroids']
            dim_latent = self.dim_latent

            # 1st Natural Parameters (with approx. uniform spacing)
            natural1, _ = _init_centroids(
                num_centroids,
                dim_latent,
                ite_max=1000,
                optimizer=lambda x: torch.optim.Adam(x, lr=1e-2),
            )

            # 2nd Natural Parameters (Vectorize Cholesky Decomposition)
            diag_idx = vector_to_tril_diag_idx(dim_latent)
            natural2_chol_vec = np.zeros((num_centroids, int(self.dim_latent * (self.dim_latent + 1) / 2)))
            natural2_chol_vec[:, diag_idx] = np.sqrt(0.5)
            natural2_chol_vec = torch.tensor(natural2_chol_vec)

            # Mixture weights
            responsibilities = torch.ones(num_centroids) / num_centroids

            # Prior
            self.prior = MixturePrior(
                responsibilities,
                natural1,
                natural2_chol_vec
            ).to(self.device.index).to(self.dtype)

    def _init_factors(self, observations):
        """ Initialize recognition network of each factor """
        if self.recognition_factors is None:
            self.recognition_factors = self._init_recognition(self.fit_params['factors_params'], observations)

    def _init_precision_factors(self):
        """ Initialize precision of each factor """

        if self.precision_chol_vec_factors is None:
            diag_idx = vector_to_tril_diag_idx(self.dim_latent)
            chol = np.zeros((self.num_factors, int(self.dim_latent * (self.dim_latent + 1) / 2)) )
            chol[:, diag_idx] = np.sqrt(0.5)
            self.precision_chol_vec_factors = fast_recognition.Precision(
                torch.tensor(chol, dtype=self.dtype)
            ).to(self.device.index)

    def _init_variational(self):
        """
        If the variational is not closed form: Initialize its recognition network.

        Notes:
            - The optimal variational first natural parameter can be expressed as a function of the first natural
            parameters of the recognition factors. We directly use them as input.

            - We empirically observed that initializing by averaging the recognition output was beneficial.
            It is therefore implemented by _pretrain_to_target
        """

        if self.recognition_variational is None and self.prior.num_centroids > 1:

            params = self.fit_params['variational_params']
            self.recognition_variational = fast_recognition.Net(
                dim_input=[self.num_factors * self.dim_latent],
                dim_latent=self.dim_latent,
                dim_hidden= params['dim_hidden'],
                non_linearity=params['non_linearity'],
                dropout=params['dropout'],
            ).to(self.device.index)

            # Transform input function
            transform = lambda x: x.reshape(x.shape[0], -1)

            # Sample input functon
            input_shape = [1000, self.num_factors, self.dim_latent]
            sample = lambda: 100 * torch.randn(input_shape, device=self.device, dtype=self.dtype)

            # Target Function
            target = lambda x: x.mean(dim=-2)

            # Pre-train the variational recognition
            _ = _pretrain_to_target(
                func=self.recognition_variational,
                func_target=target,
                func_sample=sample,
                func_transform=transform,
                stop_criterion=lambda x: x[-1] < 1e-2,
                ite_max=2000,
            )

    def _init_precision_variational(self):

        if self.precision_chol_vec_variational is None and self.prior.num_centroids > 1:

            diag_idx = vector_to_tril_diag_idx(self.dim_latent)
            chol = np.zeros((int(self.dim_latent * (self.dim_latent + 1) / 2)))
            chol[diag_idx] = np.sqrt(0.25)
            self.precision_chol_vec_variational = fast_recognition.Precision(
                torch.tensor(chol, dtype=self.dtype)
            ).to(self.device.index)

    # def _init_precision_auxiliary(self):
    #
    #     if self.precision_chol_vec_auxiliary is None:
    #
    #         diag_idx = vector_to_tril_diag_idx(self.dim_latent)
    #         chol = np.zeros((self.num_factors, int(self.dim_latent * (self.dim_latent + 1) / 2)))
    #         chol[:, diag_idx] = np.sqrt(0.5)
    #         self.precision_chol_vec_auxiliary = fast_recognition.Precision(
    #             torch.tensor(chol, dtype=self.dtype)
    #         ).to(self.device.index)
    #
    # def _init_auxiliary(self, observations):
    #     """ Initialize auxiliary recognition networks """
    #     if self.recognition_auxiliary is None:
    #         self.recognition_auxiliary = copy.deepcopy(self.recognition_factors)
    #         #self.recognition_auxiliary = self._init_recognition(self.fit_params['auxiliary_params'], observations)


def pairwise_distance(samples):

    # Number of samples
    num_samples = samples.shape[0]

    # normalize samples on the sphere
    normalized_samples = samples / torch.sqrt((samples ** 2).sum(dim=-1, keepdim=True))

    # Compute pairwise distance
    pairwise_distances = ((normalized_samples.unsqueeze(0) - normalized_samples.unsqueeze(1)) ** 2).sum(-1)

    # Fill the diagonal
    pairwise_distances_mean_tmp = pairwise_distances.sum() / (num_samples * (num_samples - 1))
    diag_idx = range(num_samples)
    pairwise_distances[diag_idx, diag_idx] = pairwise_distances_mean_tmp

    # Maximize minimal distance
    loss = - pairwise_distances.min()

    return loss, pairwise_distances, normalized_samples


def _init_centroids(
        num_centroids,
        dim_centroids,
        ite_max=10000,
        optimizer=lambda x: torch.optim.Adam(x, lr=1e-2),
):
    """Small helper to initialize mixture centroids ~ uniformly on a hypersphere"""

    if num_centroids == 1:
        samples = torch.zeros(1, dim_centroids)
        loss_tot = 0

    elif dim_centroids == 1:
        samples = torch.linspace(-1, 1, num_centroids).unsqueeze(-1)
        loss_tot = 0

    elif num_centroids > 1 and dim_centroids > 1:

        # Init Centroids
        samples_cur = torch.randn(num_centroids, dim_centroids, requires_grad=True)

        # Optimizer
        optim = optimizer([samples_cur])

        loss_tot = []
        for ite in range(ite_max):
            optim.zero_grad()
            loss, _, _ = pairwise_distance(samples_cur)
            loss.backward()
            optim.step()
            loss_tot.append(loss.item())

        # Normalize Optimal samples
        samples = samples_cur.clone().detach()
        with torch.no_grad():
            _, pairwise_distances, samples = pairwise_distance(samples)

    else:
        raise NotImplementedError()

    return samples, loss_tot


def _pretrain_to_target(
        func,
        func_target,
        func_sample,
        func_transform=lambda x: x,
        func_loss=lambda x, y: ((x - y) ** 2).mean(),
        optimizer=lambda x: torch.optim.AdamW(x, lr=5e-3, weight_decay=0.01),
        ite_max=1000,
        stop_criterion=lambda x: False,
):
    """Helper to pre-train a nn.Module() to a target"""

    # Init a buffer network
    func_buffer = copy.deepcopy(func)

    # Init Optimizer
    optim = optimizer(func_buffer.parameters())

    # Init Loss
    loss_tot = []

    # Fit
    for ite in range(ite_max):

        # Zero Past Gradients
        optim.zero_grad()

        # Sample Input
        input_cur = func_sample()

        # Target Output
        target_cur = func_target(input_cur)

        # Current Output
        output_cur = func_buffer(func_transform(input_cur))

        # Current Loss / Gradient
        loss = func_loss(target_cur, output_cur)
        loss.backward()
        optim.step()

        # Append Loss
        loss_tot.append(loss.item())

        if ite > 1:
            if stop_criterion(loss_tot):
                print('Pre-Training Completed. Stop Criterion Reached.')
                break
            elif ite == ite_max - 1:
                print('Pre-Training Completed. Stop Criterion Not satisfied.')

    func.load_state_dict(func_buffer.state_dict())

    return loss_tot

# TODO: Maybe re-instantiate this latter
# Auxiliary Network Parameters
# _default_field(self.fit_params, key='auxiliary_params', default={})
# _default_field(self.fit_params['auxiliary_params'], key='channels', default=_repeat_list((), num_factors))
# _default_field(self.fit_params['auxiliary_params'], key='kernel_conv', default=_repeat_list((), num_factors))
# _default_field(self.fit_params['auxiliary_params'], key='kernel_pool', default=_repeat_list((), num_factors))
# _default_field(self.fit_params['auxiliary_params'], key='dim_hidden', default=_repeat_list((), num_factors))
# _default_field(self.fit_params['auxiliary_params'], key='non_linearity', default=_repeat_list(F.relu, num_factors))
# _default_field(self.fit_params['auxiliary_params'], key='dropout', default=_repeat_list(0.0, num_factors))
# _default_field(self.fit_params['auxiliary_params'], key='optimizer', default=optimizer_closure_default)
# _default_field(self.fit_params['auxiliary_params'], key='scheduler', default=scheduler_closure_default)