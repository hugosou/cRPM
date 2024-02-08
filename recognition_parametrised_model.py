import torch
from torch import matmul
from flexible_multivariate_normal import (
    FlexibleMultivariateNormal,
    kl,
    flexible_kl,
    get_log_normalizer)

from typing import Union, List, Dict
from utils import optimizer_wrapper, print_loss

from prior import GPPrior
import recognition

import _initializations
import _updates

from utils import diagonalize

# TODO: minibatch
# TODO: do not forward auxiliary
# TODO: doc RPM
# TODO: If parametrized: q needs to be full covariance !!!!!
# TODO: Fit he mean of the prior: Important !
# TODO: Check Update marginals When
# TODO: modify the variational to include the means !


class RPM(_initializations.Mixin, _updates.Mixin):
    """
    Recognition Parametrised Model



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
            loss_tot: List = None,
            fit_params: Dict = None,
            prior: GPPrior = None,
            recognition_factors: recognition.Encoder = None,
            recognition_auxiliary: recognition.Encoder = None,
            recognition_variational: recognition.Encoder = None,

    ):

        # Transform Observation in list if necessary
        observations = [observations]\
            if not (type(observations) is list) and not (type(observations) is tuple) else observations

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

        with torch.no_grad():
            self._update_all(observations)
            self.loss_tot.append(self._get_loss().item())

    def _update_all(self, observations):
        self._update_factors(observations)
        self._update_variational(observations)
        self._update_marginals()
        self._update_auxiliary(observations)
        self._update_delta()

    def fit(self, observations):
        """Fit the model to the observations"""

        # Transform Observation in list if necessary
        observations = [observations]\
            if not (type(observations) is list) and not (type(observations) is tuple) else observations


        # Fit params
        fit_params = self.fit_params
        num_epoch = fit_params['num_epoch']

        # Recognition Factors Parameters
        factors_param = []
        for cur_factor in self.recognition_factors:
            factors_param += cur_factor.parameters()

        # Recognition Auxiliary Factors Parameters
        auxiliary_param = []
        for cur_factor in self.recognition_auxiliary:
            auxiliary_param += cur_factor.parameters()

        # Variational Parameters
        variational_param = self.recognition_variational.parameters()

        # Prior Parameters
        prior_param = self.prior.parameters()

        all_params = [
            [prior_param, fit_params['prior_params']],
            [factors_param, fit_params['factors_params']],
            [auxiliary_param, fit_params['auxiliary_params']],
            [variational_param, fit_params['variational_params'],]
        ]

        all_optimizers = [
            optimizer_wrapper(pp[0], pp[1]['optimizer']) for pp in all_params
        ]

        # Fit
        for epoch in range(num_epoch):

            # Forward pass
            self._update_all(observations)

            # Loss
            loss = self._get_loss()
            self.loss_tot.append(loss.item())

            # Reset Optimizers
            for opt in all_optimizers:
                opt.zero_grad()

            # Gradients
            loss.backward()

            # Gradient Steps
            for opt in all_optimizers:
                opt.step()

            # Logger
            print_loss(
                self.loss_tot[-1],
                epoch + 1,
                num_epoch,
                pct=self.fit_params['pct']
            )

    def _get_loss(self):

        # Log-Gamma term ~ J x N x T -> sum to scalar
        log_gamma = self.log_gamma.sum()

        # KL[Variational || Prior] ~ N x K or N x 1 -> sum to scalar
        KLprior = self._kl_prior().sum()

        # KL[Variational || fhat] ~ J x N x T -> sum to scalar
        KLmarginals = self._kl_marginals().sum()

        # - Loss
        normalizer = self.num_observation * self.len_observation
        free_energy = (log_gamma - KLmarginals - KLprior) / normalizer

        return - free_energy

    def _kl_marginals(self):

        # Dimension of the problem
        num_observation = self.num_observation

        # Natural Parameter from the marginal ~ 1 x N x T x K (x K)
        variational = self.dist_marginals
        variational_natural1 = variational.natural1.unsqueeze(0)
        variational_natural2 = variational.natural2.unsqueeze(0)
        variational_log_normalizer = variational.log_normalizer.unsqueeze(0)
        variational_suff_stat = [
            variational.suff_stat_mean[0].unsqueeze(0),
            variational.suff_stat_mean[1].unsqueeze(0)
        ]

        # Grasp only the m = n distribution for KL estimation
        # From J x N x N x K (x K) to J x N x K (x K)
        diag_id = range(num_observation)
        factors_delta = self.dist_delta
        diag_delta_natural1 = factors_delta.natural1[:, diag_id, diag_id]
        diag_delta_natural2 = factors_delta.natural2[:, diag_id, diag_id]
        diag_delta_factors_log_normalizer = factors_delta.log_normalizer[:, diag_id, diag_id]

        # KL(q || fhat) ~ num_factors x num_observations x len_observations
        KLmarginals = kl(
            (variational_natural1, variational_natural2),
            (diag_delta_natural1, diag_delta_natural2),
            variational_log_normalizer,
            diag_delta_factors_log_normalizer,
            variational_suff_stat
        )

        return KLmarginals

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
        prior_mean = self.prior.mean(self.inducing_locations, self.inducing_locations)
        prior_vari = self.prior.covariance(self.inducing_locations, self.inducing_locations)

        if inference_mode == 'amortized':

            # To size ~ 1 x K x M x (x M)
            prior_mean = prior_mean.unsqueeze(0)
            prior_vari = prior_vari.unsqueeze(0)
            # prior_nat1 = prior_nat1.unsqueeze(0)

            # Amortized Variational Parameters: size N x M x K ( x K)
            variational_natural = (self.dist_variational.natural1, self.dist_variational.natural2)
            variational_suffstat = self.dist_variational.suff_stat_mean
            variational_log_normalizer = self.dist_variational.log_normalizer

            if self.num_inducing_points == 1:
                # Not a time series (M = 1):
                # Reshape Prior distribution to 1 x 1 x K ( x K) by leveraging M = 1

                # Prior's Natural Parameters (reshape only works since M = 1)
                prior_mean = prior_mean.reshape(1, 1, self.dim_latent)
                prior_vard = prior_vari.reshape(1, 1, self.dim_latent)
                 # TODO Check this when M = 1
                #prior_nat1 = prior_nat1.reshape(1, 1, self.dim_latent)
                prior_nat1 = prior_mean / prior_vard
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

    def _init_all(self, observations):
        """ Init all parameters (see _initializations.Mixin) """

        # Fit parameters
        self._init_fit_params()
        self.dim_latent = self.fit_params['dim_latent']

        self._init_prior()
        self._init_factors(observations)
        self._init_auxiliary(observations)
        self._init_variational(observations)

    def _set_locations(self):
        """Handle observation and inducing Locations"""

        # Default: observation location 1D and ordered from 0 to 1
        self.observation_locations = torch.linspace(
            0, 1, self.len_observation, device=self.device, dtype=self.dtype
        ).unsqueeze(-1) if self.observation_locations is None else self.observation_locations

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

            assert self.fit_params['variational_params']['covariance'] == 'full', \
                ('Parametrized inference must use full covariance. '
                 'Fix: Modify variational_params or use amortized inference')

            assert self.len_observation > 1, \
                ('Full parametrisation not recommended for RP-FA. '
                 'Fix: Use Amortized Inference.')
        else:
            raise NotImplementedError()

