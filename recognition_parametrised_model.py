import torch
import numpy as np
from recognition import Net, MultiInputNet
from torch import matmul
from flexible_multivariate_normal import FlexibleMultivariateNormal, vector_to_tril, kl, flexible_kl, get_log_normalizer

from typing import Union, List, Dict

from utils import optimizer_wrapper, print_loss, get_minibatches

import kernels
import recognition

import _initializations

from utils import diagonalize


# TODO: Check device and dtype !
# TODO: check load and save
# TODO: when minibatching, check that we send as feww as possible.
#  If fixed variational, filter the output
# TODO: init auxiliary, update or not auxiliary. Init it to zero (default: true)
# TODO: add minibatches

class RPM(_initializations.Mixin):
    """
    Recognition Parametrised Model

    input:
        TODO: UPDATE THIS DOC
        dim_latent(tuple): dimension of latent variables
        observations(list): list of conditionally independent observations
        fit_params(dict):   dictionary of parameters for the fit
        recognition(list):  list of recognition networks
        log_prior(tensor):  log prior probabilities


    notation: for compactness, we sometimes denote:
        N: num_observations
        T: len_observations
        K: dim_latent
        M: num_inducing_points (<= T)

    """

    def __init__(
            self,
            observations: Union[torch.Tensor, List[torch.tensor]],
            observation_locations: torch.Tensor,  # len_observation x dim_locations. Location of the Observations
            inducing_locations: torch.Tensor = None,  # len_observation x dim_locations. Location of inducing points
            fit_params: Dict = None,
            loss_tot: List = None,
            prior: kernels.Kernel = None,
            recognition_factors: recognition.Encoder = None,
            recognition_auxiliary: recognition.Encoder = None,
            recognition_variational: recognition.Encoder = None,

    ):

        # Transform Observation in list if necessary
        observations = [observations] if not type(observations) is list else observations

        # Problem dimensions
        self.num_factors = len(observations)
        self.num_observation = observations[0].shape[0]
        self.len_observation = observations[0].shape[1]

        # Device and data type
        self.dtype = observations[0].dtype
        self.device = observations[0].device
        str_device = 'GPU' if torch.cuda.is_available() else 'CPU'
        print('RPM on ' + str_device + ' Observations on ' + str(self.device))

        # Set Observation and Induction Locations
        self.observation_locations = observation_locations
        self.inducing_locations = inducing_locations
        self.inducing_index = None
        self._set_locations()

        # Fit / Config parameters
        self.fit_params = fit_params

        # Init Loss
        self.loss_tot = [] if loss_tot is None else loss_tot

        # Initialize Distributions Parametrization
        self.prior = prior
        self.recognition_factors = recognition_factors
        self.recognition_auxiliary = recognition_auxiliary
        self.recognition_variational = recognition_variational
        self._init_all(observations)

        # Sanity Checks
        self._validate_model(observations)

        # Init Distributions and parameters
        self.dist_prior = None
        self.dist_delta = None
        self.dist_factors = None
        self.dist_auxiliary = None
        self.dist_marginals = None
        self.dist_variational = None
        self.log_gamma = None

        # with torch.no_grad():
        #self._update_prior()  # TODO: Check if necessary !
        self._update_factors(observations)

        self._update_variational(observations)

        self._kl_prior()

        self._update_marginals()

        self._update_auxiliary(observations)

        # self._update_variational(observations)

        # self._update_auxiliary(observations)

        # self._update_delta(observations)
        # self._update_log_gamma(observations)
        # self.loss_tot.append(self._get_loss().item())

        # TODO: update inducing points -> if inference mode ask for it only
        # TODO: update variational marginals -> depends on inference mode
        #  Amortized: pass through the network
        #  Inducing Points: compute the marginals

    def _update_auxiliary(self, observations):
        """
        Forward to auxiliary pseudo distributions
        """

        # Device and dimensions
        dtype = self.dtype
        device = self.device
        dim_latent = self.dim_latent
        num_factors = self.num_factors
        len_observation = self.len_observation
        num_observation = self.num_observation
        num_inducing_points = self.num_inducing_points

        # Prior parameters
        natural1_prior, natural2_prior = self._evaluate_prior_marginal()
        natural1_prior = natural1_prior.reshape(1, 1, 1, dim_latent)
        natural2_prior = natural2_prior.reshape(1, 1, 1, dim_latent, dim_latent)

        # Init Natural Parameters ~ J x N x K (x K)
        auxiliary1 = torch.zeros(
            num_factors,
            num_observation,
            len_observation,
            dim_latent,
            dtype=dtype,
            device=device,
        )

        auxiliary2 = torch.zeros(
            num_factors,
            num_observation,
            len_observation,
            dim_latent,
            dim_latent,
            dtype=dtype,
            device=device,
        )

        for ii, obsi in enumerate(observations):
            # Grasp current recognition
            reci = self.recognition_auxiliary[ii](obsi)

            # Temporary 1st parameter
            auxiliary1[ii] = reci[..., :dim_latent]

            # Temporary Cholesky Decomposed of 2nd parameter
            auxiliary2[ii] = vector_to_tril(reci[..., dim_latent:])

        #  We leveraged the already Stored Cholesky Decomposition (N x M x K ( x K)) in marginals
        variational_natural1 = self.dist_marginals.natural1.unsqueeze(0)
        variational_natural2_chol = self.dist_marginals.natural2_chol.unsqueeze(0)

        delta_natural1 = variational_natural1 + auxiliary1
        delta_natural2_chol = variational_natural2_chol + auxiliary2
        delta_natural2 = matmul(delta_natural2_chol, delta_natural2_chol.transpose(-1, -2))

        aux_natural1 = natural1_prior - delta_natural1
        aux_natural2 = natural2_prior - delta_natural2


    def _update_marginals(self):
        """
        Forward to marginal distribution q(z_t)
        """

        # Amortized or Parametrized variational inference
        inference_mode = self.fit_params['variational_params']['inference_mode']

        if self.num_inducing_points == 1:
            # Not a time series (M = 1) (forced amortized)
            # Marginals' size: N x 1 x K (x K)
            self.dist_marginals = self.dist_variational

        else:
            # Time series (M > 1)
            if inference_mode == 'parametrized':
                # Inducing Point's size: N x K x M (x M)
                inducing_mean, inducing_covariance = self.dist_variational.mean_covariance()

            elif inference_mode == 'amortized':
                # Inducing Point's size: N x M x K (x K)
                inducing_mean, inducing_covariance = self.dist_variational.mean_covariance()

                # Permute to N x K x M (x M)
                # Covariance is forced diagonal in K
                inducing_mean = inducing_mean.permute(0, 2, 1)
                inducing_covariance = diagonalize(
                    inducing_covariance.diagonal(dim1=-1, dim2=-2).permute(0, 2, 1)
                )

            else:
                raise NotImplementedError()

            # Inducing Points (tau ~ M) and Observations (t ~ T) # TODO: implement mini-batching
            inducing_locations = self.inducing_locations
            observation_locations = self.observation_locations

            # Kernel Posterior helpers
            K_t_t, K_tau_tau, _, K_t_tau, K_t_tau_K_tau_tau_inv = \
                self.prior.posteriors(inducing_locations, observation_locations)

            # Cov_k(t, tau) inv( Cov_k(tau, tau) ) unsqueezed to ~ 1 x K x T x 1 x M
            K_t_tau_K_tau_tau_inv = K_t_tau_K_tau_tau_inv.unsqueeze(0).unsqueeze(-2)

            # inducing_covariance - Cov_k(tau, tau) unsqueezed to ~ N x K x 1 x M x M
            delta_K = (inducing_covariance - K_tau_tau.unsqueeze(0)).unsqueeze(-3)

            # Variational Marginals Mean Reshaped and Permuted to ~ N x T x K
            marginal_mean = (
                    matmul(
                        K_t_tau_K_tau_tau_inv,
                        (inducing_mean).unsqueeze(-2).unsqueeze(-1)
                    ).squeeze(-1).squeeze(-1)
            ).permute(0, 2, 1)

            # Variational Marginals Covariance Reshaped and Permuted to ~ N x T x K (dimensions are independent)
            marginal_covariance_diag = (
                    K_t_t.unsqueeze(0)
                    + matmul(
                matmul(K_t_tau_K_tau_tau_inv, delta_K), K_t_tau_K_tau_tau_inv.transpose(-1, -2)
            ).squeeze(-1).squeeze(-1)
            ).permute(0, 2, 1)

            # Square Root and Diagonalize the marginal Covariance ~ N x T x K x K (Alternatively, use 1D MVN)
            marginal_covariance_chol = diagonalize(torch.sqrt(marginal_covariance_diag))

            # # Marginals' size: N x T x K (x K)
            # self.dist_marginals = FlexibleMultivariateNormal(
            #     marginal_mean,
            #     marginal_covariance_chol,
            #     init_natural=False,
            #     init_cholesky=True,
            #     store_suff_stat_mean=True,
            #     store_natural_chol=False,
            # )

            # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
            #  THERE MIGHT BE PLACES WHERE I TAKE THE THE INVERSE OF THE CHOL !!! BAD !

            # Use Natural parametrization
            marginal_natural2 = -0.5 * 1 / marginal_covariance_diag
            marginal_natural2_chol = diagonalize(torch.sqrt(-marginal_natural2))
            marginal_natural1 = -2 * marginal_natural2 * marginal_mean

            # Marginals' size: N x T x K (x K)
            self.dist_marginals = FlexibleMultivariateNormal(
                marginal_natural1,
                marginal_natural2_chol,
                init_natural=True,
                init_cholesky=True,
                store_suff_stat_mean=True,
                store_natural_chol=True,
            )






    def _kl_prior(self):
        """
        KL[variational || prior]

            if inference_mode == parametrised:
                prior       size:     K x M (x M)
                variational size: N x K x M (x M)
                KL div      size: N x K

            if inference_mode == amortized:
                prior       size:     K x M (x M)
                variational size: N x M x K (x K)

                if M == 1: diagonalize prior to
                    prior       size:     1 x K (x K)
                    variational size: N x 1 x K (x K)
                    KL div      size: N x 1

                if M > 1 : diagonalize variational to
                    prior       size:     K x M (x M)
                    variational size: N x K x M (x M)
                    KL div      size: N x K
        """

        # TODO: I haven't tripple proof read those KL..

        # Amortized or Parametrized variational inference
        inference_mode = self.fit_params['variational_params']['inference_mode']

        # Prior's Mean and Variance: size ~ K x M x (x M)
        prior_mean = torch.zeros(self.dim_latent, self.num_inducing_points, device=self.device, dtype=self.dtype)
        prior_nat1 = torch.zeros(self.dim_latent, self.num_inducing_points, device=self.device, dtype=self.dtype)
        prior_vari = self.prior(self.inducing_locations, self.inducing_locations)

        if inference_mode == 'amortized':

            # To size ~ 1 x K x M x (x M)
            prior_mean = prior_mean.unsqueeze(0)
            prior_nat1 = prior_nat1.unsqueeze(0)
            prior_vari = prior_vari.unsqueeze(0)

            # Amortized Variational Parameters: size N x M x K ( x K)
            variational_natural = (self.dist_variational.natural1, self.dist_variational.natural2)
            variational_suffstat = self.dist_variational.suff_stat_mean
            variational_log_normalizer = self.dist_variational.log_normalizer

            if self.num_inducing_points == 1:
                # Not a time series (M = 1):
                # Reshape Prior distribution to 1 x 1 x K ( x K) by leveraging M = 1

                # Prior's Natural Parameters (reshape only works since M = 1)
                prior_mean = prior_mean.reshape(1, 1, self.dim_latent)
                prior_nat1 = prior_nat1.reshape(1, 1, self.dim_latent)
                prior_vard = prior_vari.reshape(1, 1, self.dim_latent)
                prior_nat2 = - 0.5 * diagonalize(1 / (prior_vard + 1e-6))
                prior_natural = (prior_nat1, prior_nat2)

                # Prior Log-Normalizer
                prior_log_normalizer = get_log_normalizer(prior_mean, prior_vard, prior_natural[0])

            else:
                # Time series (M > 1)
                # Reshape variational distribution to N x K x M ( x M) by leveraging diagonal over K
                variational_nat1 = variational_natural[0].permute(0, 2, 1)
                variational_nat2 = variational_natural[1].diagonal(dim1=-1, dim2=-2).permute(0, 2, 1)
                variational_vard = - 0.5 * 1 / (variational_nat2 - 1e-6)
                variational_vari = diagonalize(variational_vard)
                variational_mean = variational_vard * variational_nat1
                variational_nat2 = diagonalize(variational_nat2)
                variational_log_normalizer = get_log_normalizer(variational_mean, variational_vard, variational_nat1)
                variational_natural = (variational_nat1, variational_nat2)
                variational_suffstat = (
                    variational_mean,
                    variational_vari + matmul(variational_mean.unsqueeze(-1), variational_mean.unsqueeze(-2))
                )

                dist_prior = FlexibleMultivariateNormal(
                    prior_mean,
                    prior_vari,
                    init_natural=False,
                    init_cholesky=False,
                )

                prior_natural = (dist_prior.natural1, dist_prior.natural2)
                prior_log_normalizer = dist_prior.log_normalizer

            # KL(q(U) || p(U)) ~ N x 1 or N x K
            KL = kl(
                variational_natural,
                prior_natural,
                variational_log_normalizer,
                prior_log_normalizer,
                variational_suffstat
            )

        elif inference_mode == 'parametrized':

            # Build Prior Distribution
            dist_prior = FlexibleMultivariateNormal(
                prior_mean,
                prior_vari,
                init_natural=False,
                init_cholesky=False,
            )

            # KL(q(U) || p(U)) ~ N x K
            KL = flexible_kl(
                self.dist_variational,
                dist_prior,
                repeat1=[0, 1],
                repeat2=[1]
            )

        else:
            raise NotImplementedError()

        return KL


    def _update_variational(self, observations):
        """
        Variational Recognition
            if amortized   : size N x T x K (x K) observation dependent
            if parametrised: size N x K x M (x M) observation independent
        """

        # Extract observations at inducing index
        obsq = [obsi[:, self.inducing_index] for obsi in observations]

        # Variational Recognition
        recq = self.recognition_variational(obsq)

        # Recognition distribution dimension
        inference_mode = self.fit_params['variational_params']['inference_mode']
        if inference_mode == 'amortized':
            dim_output = self.dim_latent
        elif inference_mode == 'parametrized':
            dim_output = self.num_inducing_points
        else:
            raise NotImplementedError()

        # Extract parameters
        natural1 = recq[..., :dim_output]
        natural2_chol = vector_to_tril(recq[..., dim_output:])

        # Build Distributions
        self.dist_variational = FlexibleMultivariateNormal(
            natural1,
            natural2_chol,
            init_natural=True,
            init_cholesky=True,
            store_natural_chol=True,
            store_suff_stat_mean=True,
        )

    def _update_prior(self):

        prior_mean = torch.zeros(self.dim_latent, self.num_inducing_points, device=self.device, dtype=self.dtype)
        prior_covariance = self.prior(self.inducing_locations, self.inducing_locations)

        self.dist_prior = FlexibleMultivariateNormal(
            prior_mean,
            prior_covariance,
            init_natural=False,
            init_cholesky=False,
        )

    def _update_factors(self, observations):

        dtype = self.dtype
        device = self.device

        dim_latent = self.dim_latent
        num_factors = self.num_factors
        len_observation = self.len_observation
        num_observation = self.num_observation

        # Init Natural Parameters ~ J x N x K (x K)
        natural1 = torch.zeros(
            num_factors,
            num_observation,
            len_observation,
            dim_latent,
            dtype=dtype,
            device=device,
        )

        natural2_chol = torch.zeros(
            num_factors,
            num_observation,
            len_observation,
            dim_latent,
            dim_latent,
            dtype=dtype,
            device=device,
        )

        for ii, obsi in enumerate(observations):
            # Grasp current recognition
            reci = self.recognition_factors[ii](obsi)

            # 1st Natural Parameter
            natural1[ii] = reci[..., :dim_latent]

            # Cholesky Decomposition of (minus) the second natural parameter
            natural2_chol[ii] = vector_to_tril(reci[..., dim_latent:])

        # Prior parameters
        natural1_prior, natural2_prior = self._evaluate_prior_marginal()
        natural1_prior = natural1_prior.reshape(1, 1, 1, dim_latent)
        natural2_prior = natural2_prior.reshape(1, 1, 1, dim_latent, dim_latent)

        # Build factor distributions
        natural1 = natural1_prior + natural1
        natural2 = natural2_prior + matmul(-natural2_chol, natural2_chol.transpose(-1, -2))

        # Build Distributions
        self.dist_factors = FlexibleMultivariateNormal(
            natural1,
            natural2,
            init_natural=True,
            init_cholesky=False,
            store_natural_chol=True,
            store_suff_stat_mean=True,  # TODO: check if this is really needed  ?
        )

    def _evaluate_prior_marginal(self):
        """
        Marginalize prior over time
        Implicit assumption (!): the kernel is stationary and its mean is zero
        """
        dim_latent = self.dim_latent

        natural1_prior = torch.zeros(dim_latent, dtype=self.dtype, device=self.device)
        natural2_prior = - 0.5 * diagonalize(
            1 / self.prior(self.inducing_locations[:1], self.inducing_locations[:1]).squeeze(dim=(-1, -2))
            )

        return natural1_prior, natural2_prior

    def _update_delta(self):
        raise NotImplementedError()



    def _update_log_gamma(self):
        raise NotImplementedError()

    def _init_all(self, observations):
        """ Init all parameters (see _initializations.Mixin) """

        # Fit parameters
        self._init_fit_params()
        self.dim_latent = self.fit_params['dim_latent']

        self._init_prior()
        self._init_factors(observations)
        self._init_auxiliary(observations)
        self._init_variational(observations)

    def fit(self, observations):
        """Fit the model to the observations"""

        # Fit params
        fit_params = self.fit_params
        num_epoch = fit_params['num_epoch']
        mini_batches = self.mini_batches

        # Network Optimizers
        factors_param = []
        for cur_factor in self.recognition_factors:
            factors_param += cur_factor.parameters()
        optimizer_factors = optimizer_wrapper(factors_param, fit_params['optimizer_factors'])

        variational_param = self.recognition_variational.parameters()
        optimizer_variational = optimizer_wrapper(variational_param, fit_params['optimizer_variational'])

        # Fit
        for epoch in range(num_epoch):

            # Current epoch losses
            loss_minibatch = []
            num_minibatch = len(mini_batches[epoch])

            for batch_id in range(num_minibatch):
                # Set current minibatch
                self.epoch_batch = [epoch, batch_id]

                # Forward pass
                self._update_factors(observations)
                self._update_variational(observations)
                self._update_factors_delta()

                # Loss
                loss = self._get_loss()
                loss_minibatch.append(loss.item())

                # Step
                optimizer_factors.zero_grad()
                optimizer_variational.zero_grad()
                loss.backward()
                optimizer_factors.step()
                optimizer_variational.step()

            # Gather loss
            self.loss_tot.append(np.mean(loss_minibatch))

            # Logger
            print_loss(self.loss_tot[-1], epoch + 1, num_epoch, pct=self.fit_params['pct'])

    def _get_loss(self):

        # Dimension of the problem
        num_observation = self.num_observation
        num_observation = len(self.mini_batches[self.epoch_batch[0]][self.epoch_batch[1]])

        # log gamma term
        log_gamma = self.log_gamma

        # Natural Parameter from the marginal
        variational = self.variational
        variational_natural1 = variational.natural1.unsqueeze(0)
        variational_natural2 = variational.natural2.unsqueeze(0)
        variational_log_normalizer = variational.log_normalizer.unsqueeze(0)
        variational_suff_stat = [variational.suff_stat_mean[0].unsqueeze(0),
                                 variational.suff_stat_mean[1].unsqueeze(0)]

        # Grasp only the m = n distribution for KL estimation
        diag_id = range(num_observation)
        factors_delta = self.factors_delta
        diag_delta_natural1 = factors_delta.natural1[:, diag_id, diag_id]
        diag_delta_natural2 = factors_delta.natural2[:, diag_id, diag_id]
        diag_delta_factors_log_normalizer = factors_delta.log_normalizer[:, diag_id, diag_id]

        # KL(q || fhat) ~ num_factors x num_observations x len_observations
        KLqfhat = kl((variational_natural1, variational_natural2),
                     (diag_delta_natural1, diag_delta_natural2),
                     variational_log_normalizer, diag_delta_factors_log_normalizer,
                     variational_suff_stat)

        # KL[variational, prior] for Inducing points ~ num_observations x dim_latent
        prior = self.prior
        KLqp = flexible_kl(variational, prior, repeat1=[0], repeat2=[])

        # - Loss
        free_energy = (log_gamma.sum() - KLqfhat.sum() - KLqp.sum()) / num_observation

        return - free_energy

    def _set_locations(self):
        """Handle observation and inducing Locations"""

        # Default: observation location 1D and ordered from 0 to 1
        self.observation_locations = torch.linspace(0, 1, self.len_observation).unsqueeze(-1) \
            if self.observation_locations is None else self.observation_locations

        # Default: inducing points at every observed location
        self.inducing_locations = self.observation_locations \
            if self.inducing_locations is None else self.inducing_locations

        # Number of Inducting Points
        self.num_inducing_points = self.inducing_locations.shape[0]

        # Distance: Observation - Inducing Locations
        ind_dist = ((self.inducing_locations.unsqueeze(0) - self.observation_locations.unsqueeze(1)) ** 2).sum(-1)
        _, loc_min = ind_dist.min(dim=0)

        # For each inducing point, store closest observed location
        self.inducing_index = loc_min

    def _validate_model(self, observations):
        """A range of Sanity Checks"""

        # Consistent observations shape
        assert all([i.shape[0] == self.num_observation for i in observations]), "Inconsistent number of observations"
        assert all([i.shape[1] == self.len_observation for i in observations]), "Inconsistent length of observations"

        # Some modeling choices might conflict
        inference_mode = self.fit_params['variational_params']['inference_mode']
        if inference_mode == 'amortized':

            # Amortized Time Series Modeling
            if self.len_observation > 1:

                # Distance between observations and inducing locations
                loc_ind = self.inducing_locations
                loc_obs = self.observation_locations[self.inducing_index]
                loc_dist = ((loc_ind - loc_obs) ** 2).sum(dim=-1).max()

                assert loc_dist < 1e-6, \
                    ('Each amortised Inducing Locations should match one observation. '
                     'Fix: Redefine Locations or use parametrized inference')

                assert 'diag' in self.fit_params['variational_params']['covariance'], \
                    ('Full Covariance incompatible with Amortised RP-GPFA.  '
                     'Fix: Use diagonal covariance or parametrized inference.')

        elif inference_mode == 'parametrized':

            assert self.len_observation > 1, \
                ('Full parametrisation not recommended for RP-FA. '
                 'Fix: Use Amortized Inference.')
        else:
            raise NotImplementedError()

    #
    # def _update_factors_delta(self):
    #
    #     # Natural Parameters of the factors ~ J x 1 x N x K (x K)
    #     factors_natural1 = self.factors.natural1.unsqueeze(1)
    #     factors_natural2 = self.factors.natural2.unsqueeze(1)
    #     factors_log_normaliser = self.factors.log_normalizer.unsqueeze(1)
    #
    #     variational_natural1 = self.variational.natural1
    #     variational_natural2 = self.variational.natural2
    #
    #     prior_natural1 = self.prior.natural1
    #     prior_natural2 = self.prior.natural2
    #
    #     auxiliary_natural1 = (prior_natural1 - variational_natural1).unsqueeze(0).unsqueeze(2)
    #     auxiliary_natural2 = (prior_natural2 - variational_natural2).unsqueeze(0).unsqueeze(2)
    #
    #     # eta_m - eta_tilde_n ~ J x N x N x K (x K)
    #     delta_natural1 = factors_natural1 - auxiliary_natural1
    #     delta_natural2 = factors_natural2 - auxiliary_natural2
    #
    #     self.factors_delta = FlexibleMultivariateNormal(delta_natural1, delta_natural2,
    #                                                     init_natural=True, init_cholesky=False,
    #                                                     store_suff_stat_mean=True)
    #
    #     # Ratio of log-nomaliser differences ~ J x N x N
    #     delta_log_normalizer = self.factors_delta.log_normalizer - factors_log_normaliser
    #     log_weights = delta_log_normalizer - torch.logsumexp(delta_log_normalizer, dim=2, keepdim=True)
    #     self.log_gamma = log_weights.diagonal(dim1=1, dim2=2)

    # def _update_variational(self, observations, full=False):
    #
    #     # Latent Dimensions
    #     dim_latent = self.dim_latent
    #
    #     # Grasp current batch
    #     epoch_batch = self.epoch_batch
    #     batch_id = self.mini_batches[epoch_batch[0]][epoch_batch[1]]
    #     num_observation = len(batch_id)
    #     full = True if num_observation == self.num_observation else full
    #     observation_cur = [obsi if full else obsi[batch_id] for obsi in observations]
    #
    #     # Get natural parameters using recognition network
    #     naturals = self.recognition_variational(observation_cur)
    #
    #     # 1st Natural Parameter
    #     natural1 = naturals[..., :dim_latent]
    #
    #     # Cholesky Decomposition of the (-) second natural parameter
    #     natural2_chol = vector_to_tril(naturals[..., dim_latent:])
    #
    #     # Build variational distributions
    #     self.variational = FlexibleMultivariateNormal(natural1, natural2_chol,
    #                                                   init_natural=True, init_cholesky=True,
    #                                                   store_natural_chol=True, store_suff_stat_mean=True)
