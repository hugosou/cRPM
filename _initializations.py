# Imports
import torch
import numpy as np
import torch.nn.functional as F
from torch.linalg import cholesky

import kernels
import recognition
from utils import diagonalize
from flexible_multivariate_normal import tril_to_vector


class Mixin:
    """
        Mixin class containing necessary methods for initializing RPGPFA model
    """

    def _init_fit_params(self):
        """ Default Fit parameters """

        # Init dictionary
        if self.fit_params is None:
            self.fit_params = {}

        # Latent dimensions
        if not ('dim_latent' in self.fit_params.keys()):
            self.fit_params['dim_latent'] = 1

        # Prior
        if not ('prior_params' in self.fit_params.keys()):
            self.fit_params['prior_params'] = {}

        # Prior Gaussian Process Covariance Kernel Type
        if not ('gp_kernel' in self.fit_params['prior_params'].keys()):
            self.fit_params['prior_params']['gp_kernel'] = 'RBF'

        # Prior Gaussian Process Covariance Scale
        if not ('fit_kernel_scale' in self.fit_params['prior_params'].keys()):
            self.fit_params['prior_params']['fit_kernel_scale'] = False

        # Prior Gaussian Process Covariance LengthScale
        if not ('fit_kernel_lengthscale' in self.fit_params['prior_params'].keys()):
            self.fit_params['prior_params']['fit_kernel_lengthscale'] = True

        # Prior Scale
        if not ('scale' in self.fit_params['prior_params'].keys()):
            self.fit_params['prior_params']['scale'] = 1.0

        # Prior LengthScale
        if not ('lengthscale' in self.fit_params['prior_params'].keys()):
            self.fit_params['prior_params']['lengthscale'] = 0.1

        # Recognition Factors
        if not ('factors_params' in self.fit_params.keys()):
            self.fit_params['factors_params'] = {}

        # Factors Channels
        if not ('channels' in self.fit_params['factors_params'].keys()):
            self.fit_params['factors_params']['channels'] = [
                () for _ in range(self.num_factors)
            ]

        # Factors 2D Convolutions
        if not ('kernel_conv' in self.fit_params['factors_params'].keys()):
            self.fit_params['factors_params']['kernel_conv'] = [
                () for _ in range(self.num_factors)
            ]

        # Factors 2D Pooling
        if not ('kernel_pool' in self.fit_params['factors_params'].keys()):
            self.fit_params['factors_params']['kernel_pool'] = [
                () for _ in range(self.num_factors)
            ]

        # Factors MLP Layers
        if not ('dim_hidden' in self.fit_params['factors_params'].keys()):
            self.fit_params['factors_params']['dim_hidden'] = [
                () for _ in range(self.num_factors)
            ]

        # Factors Activation function
        if not ('nonlinearity' in self.fit_params['factors_params'].keys()):
            self.fit_params['factors_params']['nonlinearity'] = [
                F.relu for _ in range(self.num_factors)
            ]

        # Factors Covariance Constraint
        if not ('covariance' in self.fit_params['factors_params'].keys()):
            self.fit_params['factors_params']['covariance'] = [
                'fixed' for _ in range(self.num_factors)
            ]

        # Recognition Auxiliary Factors
        if not ('auxiliary_params' in self.fit_params.keys()):
            self.fit_params['auxiliary_params'] = {}

        # Auxiliary Factors Channels
        if not ('channels' in self.fit_params['auxiliary_params'].keys()):
            self.fit_params['auxiliary_params']['channels'] = [
                () for _ in range(self.num_factors)
            ]

        # Auxiliary Factors 2D Convolutions
        if not ('kernel_conv' in self.fit_params['auxiliary_params'].keys()):
            self.fit_params['auxiliary_params']['kernel_conv'] = [
                () for _ in range(self.num_factors)
            ]

        # Auxiliary Factors 2D Pooling
        if not ('kernel_pool' in self.fit_params['auxiliary_params'].keys()):
            self.fit_params['auxiliary_params']['kernel_pool'] = [
                () for _ in range(self.num_factors)
            ]

        # Auxiliary Factors MLP Layers
        if not ('dim_hidden' in self.fit_params['auxiliary_params'].keys()):
            self.fit_params['auxiliary_params']['dim_hidden'] = [
                () for _ in range(self.num_factors)
            ]

        # Auxiliary Factors Activation function
        if not ('nonlinearity' in self.fit_params['auxiliary_params'].keys()):
            self.fit_params['auxiliary_params']['nonlinearity'] = [
                F.relu for _ in range(self.num_factors)
            ]

        # Auxiliary Factors Covariance Constraint
        if not ('covariance' in self.fit_params['auxiliary_params'].keys()):
            self.fit_params['auxiliary_params']['covariance'] = [
                'fixed' for _ in range(self.num_factors)
            ]

        # Variational
        if not ('variational_params' in self.fit_params.keys()):
            self.fit_params['variational_params'] = {}

        if not ('inference_mode' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['inference_mode'] = 'amortized'

        # Variational Channels
        if not ('channels' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['channels'] = [
                () for _ in range(self.num_factors)
            ]

        # Variational 2D Convolutions
        if not ('kernel_conv' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['kernel_conv'] = [
                () for _ in range(self.num_factors)
            ]

        # Variational 2D Pooling
        if not ('kernel_pool' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['kernel_pool'] = [
                () for _ in range(self.num_factors)
            ]

        # Variational MLP Layers
        if not ('dim_hidden' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['dim_hidden'] = [
                () for _ in range(self.num_factors)
            ]

        # Variational Merged MLP Layers
        if not ('dim_hidden_merged' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['dim_hidden_merged'] = ()

        # Variational Factors Activation function
        if not ('nonlinearity' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['nonlinearity'] = [
                F.relu for _ in range(self.num_factors)
            ]

        # Variational Factors Activation function (post merge)
        if not ('nonlinearity_merged' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['nonlinearity_merged'] = F.relu

        # Variational Factors Covariance Constraint
        if not ('covariance' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['covariance'] = 'fixed'

        # Iterations
        if not ('num_epoch' in self.fit_params.keys()):
            self.fit_params['num_epoch'] = 500

        # Logger
        if not ('pct' in self.fit_params.keys()):
            self.fit_params['pct'] = 0.01

        # Ergodic assumption
        if not ('ergodic' in self.fit_params.keys()):
            self.fit_params['ergodic'] = False

        # Default Optimizers
        if not ('optimizer' in self.fit_params['prior_params'].keys()):
            self.fit_params['prior_params']['optimizer'] = {'name': 'Adam', 'param': {'lr': 1e-3}}
        if not ('optimizer' in self.fit_params['factors_params'].keys()):
            self.fit_params['factors_params']['optimizer'] = {'name': 'Adam', 'param': {'lr': 1e-3}}
        if not ('optimizer' in self.fit_params['auxiliary_params'].keys()):
            self.fit_params['auxiliary_params']['optimizer'] = {'name': 'Adam', 'param': {'lr': 1e-4}}
        if not ('optimizer' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['optimizer'] = {'name': 'Adam', 'param': {'lr': 1e-4}}

        # Dropout
        if not ('dropout' in self.fit_params['factors_params'].keys()):
            self.fit_params['factors_params']['dropout'] = 0.0
        if not ('dropout' in self.fit_params['auxiliary_params'].keys()):
            self.fit_params['auxiliary_params']['dropout'] = 0.0
        if not ('dropout' in self.fit_params['variational_params'].keys()):
            self.fit_params['variational_params']['dropout'] = 0.0

    def _init_prior(self):
        """ Initialise parameters of k=1..K independent kernels """

        if self.prior is None:

            # Number of GP prior
            dim_latent = self.dim_latent

            # Kernel Type
            kernel_type = self.fit_params['prior_params']['gp_kernel']

            # Fitted Parameters
            fit_lengthscale = self.fit_params['prior_params']['fit_kernel_lengthscale']
            fit_scale = self.fit_params['prior_params']['fit_kernel_scale']

            # (Length)scales
            scale0 = self.fit_params['prior_params']['scale']
            lengthscale0 = self.fit_params['prior_params']['lengthscale']


            scale = scale0 * torch.ones(dim_latent)
            lengthscale = lengthscale0 * torch.ones(dim_latent)

            if kernel_type == 'RBF':
                self.prior = kernels.RBFKernel(
                    scale,
                    lengthscale,
                    fit_scale=fit_scale,
                    fit_lengthscale=fit_lengthscale
                ).to(self.device.index)
            else:
                raise NotImplementedError()

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
                zero_init=True,
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
                prior_covariance = self.prior(self.inducing_locations, self.inducing_locations).detach().to('cpu')

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

            # In this case, the covariance is temporal (!)
            recognition_variational = recognition.FullyParametrised(
                num_inducing_points,
                [num_observation, dim_latent],
                covariance=covariance,
            ).to(self.device.index)


        else:
            raise NotImplementedError()

        self.recognition_variational = recognition_variational








    # def _init_prior_mean_param(self):
    #     """Initialize the mean parametrization of k=1..K independent prior Gaussian Processes"""
    #
    #     fit_prior_mean = self.fit_params['fit_prior_mean']
    #     dim_latent = self.fit_params['dim_latent']
    #     num_inducing_point = self.num_inducing_point
    #
    #     # Mean vector ~ dim_latent x num_inducing
    #     prior_mean_param_tmp = np.zeros((dim_latent, num_inducing_point))
    #     prior_mean_param = torch.tensor(prior_mean_param_tmp, device=self.device, dtype=self.dtype,
    #                                     requires_grad=fit_prior_mean)
    #
    #     # The scale is fixed
    #     scale_tmp = np.ones(dim_latent)
    #     scale = torch.tensor(scale_tmp, dtype=self.dtype, device=self.device, requires_grad=False)
    #
    #     # Lengthscale
    #     lengthscale_tmp = 0.02 * np.ones(dim_latent)
    #     lengthscale = torch.tensor(lengthscale_tmp, dtype=self.dtype, device=self.device, requires_grad=fit_prior_mean)
    #
    #     self.prior_mean_param = (prior_mean_param, scale, lengthscale)
    #
    # def _init_kernel(self):
    #     """ Initialise parameters of k=1..K independent kernels """
    #
    #     # Number of GP prior
    #     dim_latent = self.dim_latent
    #
    #     # Grasp Kernel Type
    #     kernel_name = self.fit_params['gp_kernel']
    #
    #     dtype = self.dtype
    #     device = self.device
    #
    #     copy_scale = False
    #
    #     # (Length)scales
    #     scale = 1 * torch.ones(dim_latent, dtype=dtype, device=device, requires_grad=False)
    #     lengthscale = 0.01 * torch.ones(dim_latent, dtype=dtype, device=device, requires_grad=False)
    #
    #     if kernel_name == 'RBF':
    #         self.prior_covariance_kernel = kernels.RBFKernel(scale, lengthscale, copy_scale=copy_scale)
    #
    #     elif kernel_name == 'RQ':
    #         alpha = torch.ones(dim_latent, dtype=dtype, device=device, requires_grad=False)
    #         self.prior_covariance_kernel = kernels.RQKernel(scale, lengthscale, alpha, copy_scale=copy_scale)
    #
    #     elif 'Matern' in kernel_name:
    #         nu = int(kernel_name[-2]) / int(kernel_name[-1])
    #         self.prior_covariance_kernel = kernels.MaternKernel(scale, lengthscale, nu, copy_scale=copy_scale)
    #
    #     elif kernel_name == 'Periodic':
    #         period = 0.1 * torch.ones(dim_latent, dtype=dtype, device=device, requires_grad=False)
    #         self.prior_covariance_kernel = kernels.PeriodicKernel(scale, lengthscale, period, copy_scale=copy_scale)
    #
    #     else:
    #         raise NotImplementedError()
    #



    #
    # def _init_inducing_points(self):
    #     """ Initialise the inducing points variational distribution """
    #
    #     # Setting and dimensions
    #     dtype = self.dtype
    #     device = self.device
    #     dim_latent = self.dim_latent
    #     num_observation = self.num_observation
    #     num_inducing_point = self.num_inducing_point
    #     inducing_locations = self.inducing_locations
    #
    #     # 1st Natural Parameter
    #     natural1_tmp = torch.zeros(num_observation, dim_latent, num_inducing_point, dtype=dtype, device=device)
    #     natural1 = natural1_tmp.clone().detach().requires_grad_(True)
    #
    #     # 2nd Natural Parameter is initialized using prior
    #     prior_covariance = self.prior_covariance_kernel(inducing_locations, inducing_locations).detach().clone()
    #     covariance_tmp = prior_covariance.unsqueeze(0).repeat(self.num_observation, 1, 1, 1)
    #     Id = diagonalize(torch.ones(num_observation, dim_latent, num_inducing_point, dtype=dtype, device=device))
    #     natural2_chol = cholesky(0.5 * torch.linalg.inv(covariance_tmp + 1e-3 * Id))
    #     natural2_chol_vec = tril_to_vector(natural2_chol).clone().detach().requires_grad_(True)
    #
    #     self.inducing_points_param = (natural1, natural2_chol_vec)
