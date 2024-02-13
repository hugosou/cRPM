# Imports
import torch
import numpy as np
import torch.nn.functional as F
from torch.linalg import cholesky

import kernels
import recognition
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
        _default_field(self.fit_params['auxiliary_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['auxiliary_params'], key='scheduler', default=scheduler_closure_default)

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
        _default_field(self.fit_params['variational_params'], key='nonlinearity_merged', default=_repeat_list(F.relu, num_factors))
        _default_field(self.fit_params['variational_params'], key='covariance', default=_repeat_list('fixed', num_factors))
        _default_field(self.fit_params['variational_params'], key='dropout', default=0.0)
        _default_field(self.fit_params['variational_params'], key='optimizer', default=optimizer_closure_default)
        _default_field(self.fit_params['variational_params'], key='scheduler', default=scheduler_closure_default)

    def _init_prior(self):
        """ Initialise parameters of k=1..K independent kernels """

        if self.prior is None:

            # Number of GP prior and IP
            dim_latent = self.dim_latent
            num_inducing_points = self.num_inducing_points

            # Prior Parameters
            params = self.fit_params['prior_params']

            # Kernel Type
            kernel_type = params['gp_kernel']

            # Prior's Mean params
            mean0 = [
                torch.zeros(dim_latent, num_inducing_points),
                params['fit_prior_mean_param'],
            ]

            # Prior's Mean scale param
            scale0 = [
                params['scale_prior'] * torch.ones(dim_latent),
                params['fit_kernel_scale_prior'],
            ]

            # Prior's Mean lengthscale param
            lengthscale0 = [
                params['lengthscale_prior'] * torch.ones(dim_latent),
                params['fit_kernel_lengthscale_prior'],
            ]

            # Prior Scale Parameters
            scale1 = [
                params['scale'] * torch.ones(dim_latent),
                params['fit_kernel_scale'],
            ]

            # Prior Lengthscale Parameters
            lengthscale1 = [
                params['lengthscale'] * torch.ones(dim_latent),
                params['fit_kernel_lengthscale'],
            ]

            self.prior = GPPrior(
                mean0,
                scale0,
                scale1,
                lengthscale0,
                lengthscale1,
                covariance_type0=kernel_type,
                covariance_type1=kernel_type,
            ).to(self.device.index)

    def _init_factors(self, observations):
        """ Initialize recognition network of each factor """

        dim_inputs = [
            obsi.shape[2:] for obsi in observations
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
            neti = recognition.Net(
                dim_input=dim_inputs[obsi],
                dim_latent=dim_latent,
                covariance=covariance[obsi],
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

        dim_inputs = [
            obsi.shape[2:] for obsi in observations
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

        # Covariance type
        covariance = fit_params["covariance"]

        # Build and Append networks
        recognition_auxiliary = []
        for obsi in range(num_factors):
            neti = recognition.Net(
                dim_input=dim_inputs[obsi],
                dim_latent=dim_latent,
                covariance=covariance[obsi],
                kernel_conv=kernel_conv[obsi],
                kernel_pool=kernel_pool[obsi],
                channels=channels[obsi],
                dim_hidden=dim_hidden[obsi],
                non_linearity=non_linearity[obsi],
                zero_init=False, # TODO INIT TO ZERO MAYBE ?
                dropout=dropout,
            ).to(self.device.index)
            recognition_auxiliary.append(neti)

        self.recognition_auxiliary = recognition_auxiliary

    def _init_variational(self, observations):

        # Init Recognition Network
        dim_inputs = [
            obsi.shape[2:] for obsi in observations
        ]

        # Grasp fit params
        fit_params = self.fit_params['variational_params']
        dim_latent = self.dim_latent

        if fit_params['inference_mode'] == 'amortized':

            # Convolutional parameters
            channels = fit_params["channels"]
            kernel_conv = fit_params["kernel_conv"]
            kernel_pool = fit_params["kernel_pool"]
            dropout = fit_params["dropout"]

            # Covariance type
            covariance = fit_params["covariance"]

            # Fully connected layers parameters
            dim_hidden = fit_params["dim_hidden"]
            dim_hidden_merged = fit_params["dim_hidden_merged"]
            non_linearity = fit_params["nonlinearity"]
            non_linearity_merged = fit_params["nonlinearity_merged"]


            recognition_variational = recognition.MultiInputNet(
                dim_inputs,
                dim_latent,
                covariance=covariance,
                kernel_conv=kernel_conv,
                kernel_pool=kernel_pool,
                channels=channels,
                dim_hidden=dim_hidden,
                dim_hidden_merged=dim_hidden_merged,
                non_linearity= non_linearity,
                non_linearity_merged=non_linearity_merged,
                zero_init=False,
                dropout=dropout,
            ).to(self.device.index)

        elif fit_params['inference_mode'] == 'parametrized':

            # Problem dimension
            dim_latent = self.dim_latent
            num_observation = self.num_observation
            num_inducing_points = self.num_inducing_points

            # Covariance type
            covariance = fit_params["covariance"]

            # GP Prior Covariances
            with torch.no_grad():
                prior_covariance = self.prior.covariance(
                    self.inducing_locations,
                    self.inducing_locations
                ).detach().to('cpu')

            # delta to avoid inversion issues
            Id = 1e-3 * torch.eye(
                self.num_inducing_points,
                dtype=prior_covariance.dtype,
            )

            # Second natural parameters
            natural2_chol = torch.linalg.cholesky(0.5 * torch.linalg.inv(prior_covariance + Id))
            natural2_vect = tril_to_vector(natural2_chol).unsqueeze(0).repeat(self.num_observation, 1, 1)

            # 1st natural parameters
            natural1 = torch.zeros(
                self.num_observation,
                self.dim_latent,
                self.num_inducing_points,
                dtype=prior_covariance.dtype,
            )

            # In this case, the covariance is temporal (!)
            recognition_variational = recognition.FullyParametrised(
                num_inducing_points,
                [num_observation, dim_latent],
                covariance=covariance,
                init=(natural1, natural2_vect),
            ).to(self.device.index)

        else:
            raise NotImplementedError()

        self.recognition_variational = recognition_variational


