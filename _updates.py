import torch
from torch import matmul
from flexible_multivariate_normal import (
    NNFlexibleMultivariateNormal,
    FlexibleMultivariateNormal,
    vector_to_tril,
)

from utils import diagonalize


class Mixin:
    """
        Mixin class containing necessary methods for updating RPM model distributions
    """

    def _update_delta(self):
        """
        Build all the ration distributions (factors - factors_tilde)
        """

        # Natural Parameters of the factors ~ J x 1 x N x T x K (x K)
        factors_natural1 = self.dist_factors.natural1.unsqueeze(1)
        factors_natural2 = self.dist_factors.natural2.unsqueeze(1)
        factors_log_normaliser = self.dist_factors.log_normalizer.unsqueeze(1)

        # Pseudo Natural Parameters of the auxiliary factors ~ J x N x 1 x T x K (x K)
        factors_tilde_natural1 = self.dist_auxiliary.natural1.unsqueeze(2)
        factors_tilde_natural2 = self.dist_auxiliary.natural2.unsqueeze(2)

        # eta_m - eta_tilde_n ~ J x N x N x T x K (x K)
        delta_natural1 = factors_natural1 - factors_tilde_natural1
        delta_natural2 = factors_natural2 - factors_tilde_natural2

        # fhat in the paper
        self.dist_delta = FlexibleMultivariateNormal(
            delta_natural1,
            delta_natural2,
            init_natural=True,
            init_cholesky=False,
            store_suff_stat_mean=True
        )

        # Ratio of log-nomaliser differences ~ J x N x N x T
        delta_log_normalizer = self.dist_delta.log_normalizer - factors_log_normaliser

        # In the ergodic cae, the sum is over T and N
        if self.fit_params['ergodic']:
            log_weights = delta_log_normalizer - torch.logsumexp(delta_log_normalizer, dim=(2, 3), keepdim=True)
        else:
            log_weights = delta_log_normalizer - torch.logsumexp(delta_log_normalizer, dim=2, keepdim=True)

        # log Gamma ~ J x N x T
        self.log_gamma = log_weights.diagonal(dim1=1, dim2=2).permute(0, 2, 1)

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

        # Prior parameters
        natural1_prior, natural2_prior = self._evaluate_prior_marginal()
        natural1_prior = natural1_prior.reshape(1, 1, len_observation, dim_latent)
        natural2_prior = diagonalize(natural2_prior.reshape(1, 1, len_observation, dim_latent))

        # Init Natural Parameters ~ J x N x T x K (x K)
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

        d_natural1 = variational_natural1 + auxiliary1
        d_natural2_chol = variational_natural2_chol + auxiliary2
        d_natural2 = -matmul(d_natural2_chol, d_natural2_chol.transpose(-1, -2))

        aux_natural1 = natural1_prior - d_natural1
        aux_natural2 = natural2_prior - d_natural2

        self.dist_auxiliary = NNFlexibleMultivariateNormal(
            aux_natural1,
            aux_natural2,
        )

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

            # Prior Mean ~ 1 x K x T
            prior_mean_t = self.prior.mean(observation_locations, inducing_locations).unsqueeze(0)

            # Prior Mean ~ 1 x K x M
            prior_mean_tau = self.prior.mean(inducing_locations, inducing_locations).unsqueeze(0)

            # Kernel Posterior helpers
            K_t_t, K_tau_tau, _, K_t_tau, K_t_tau_K_tau_tau_inv = \
                self.prior.covariance.posteriors(inducing_locations, observation_locations)

            # Cov_k(t, tau) inv( Cov_k(tau, tau) ) unsqueezed to ~ 1 x K x T x 1 x M
            K_t_tau_K_tau_tau_inv = K_t_tau_K_tau_tau_inv.unsqueeze(0).unsqueeze(-2)

            # inducing_covariance - Cov_k(tau, tau) unsqueezed to ~ N x K x 1 x M x M
            delta_K = (inducing_covariance - K_tau_tau.unsqueeze(0)).unsqueeze(-3)

            # Variational Marginals Mean Reshaped and Permuted to ~ N x T x K
            marginal_mean = (
                prior_mean_t + matmul(
                    K_t_tau_K_tau_tau_inv,
                    (inducing_mean - prior_mean_tau).unsqueeze(-2).unsqueeze(-1)
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

        # Prior parameters ~ 1 x 1 x T x K (x K)
        natural1_prior, natural2_prior = self._evaluate_prior_marginal()
        natural1_prior = natural1_prior.reshape(1, 1, len_observation, dim_latent)
        natural2_prior = diagonalize(natural2_prior.reshape(1, 1, len_observation, dim_latent))

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
        -> The mean should not be zero
        T x K
        """


        prior_mean = self.prior.mean(
            self.observation_locations,
            self.inducing_locations
        )

        prior_cova_diag = self.prior.covariance(
            self.observation_locations,
            self.observation_locations
        ).diagonal(dim1=-1, dim2=-2)

        natural2_prior = - 0.5 / (1e-6 + prior_cova_diag)
        natural1_prior = - 2 * natural2_prior * prior_mean

        return natural1_prior.permute(1, 0), natural2_prior.permute(1, 0)
