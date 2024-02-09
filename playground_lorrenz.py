import torch
import numpy as np
from torch import matmul
import matplotlib.pyplot as plt

from utils_generate_toydatasets import generate_lorenz
from kernels import RBFKernel

import torch
import numpy as np
from torch import matmul
import matplotlib.pyplot as plt
from kernels import RBFKernel
import torch.nn.functional as F
from utils_generate_toydatasets import generate_lorenz
from mpl_toolkits.mplot3d import Axes3D



#%%
# Reproducibility
np.random.seed(10)
torch.manual_seed(10)

# GPUs ?
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dimension of the problem
dim_observations = 100
num_observations = 10
len_observations = 100
num_inducing_points = 25
dim_latent_true = 3

# Generate Lorenz Dynamics
dtt_simulation = 0.001   # a.u
len_simulation = 3e3
num_simulation = num_observations
dim_simulation = 3
init_simulation = np.array([2.3274,  3.8649, 18.2295])
vari_simulation = 0.1

# Normalize Lorenz Dynamics to [-1, 1]
lorenz_raw = torch.tensor(generate_lorenz(num_simulation, int(len_simulation) -1, dtt_simulation, init_simulation, vari_simulation), dtype=dtype)
lorenz_nor = lorenz_raw.reshape(lorenz_raw.shape[0] * lorenz_raw.shape[1], dim_simulation)
lorenz_nor -= lorenz_nor.min(dim=0, keepdim=True)[0]
lorenz_nor /= lorenz_nor.max(dim=0, keepdim=True)[0]
lorenz_nor = 2 * lorenz_nor - 1

# Reshape Dynamics
lorenz_nor = lorenz_nor.reshape(num_simulation, int(len_simulation), dim_simulation)
time_idx = np.linspace(0, len_simulation-1, len_observations).round().astype(int)
lorenz_nor = lorenz_nor[:, time_idx]

# Add Gaussian Noise To each trajectories
noise_kernel = RBFKernel(0.1 * torch.ones(dim_simulation), 0.1 * torch.ones(dim_simulation))
KK = noise_kernel(torch.linspace(0, 1, len_observations).unsqueeze(-1), torch.linspace(0, 1, len_observations).unsqueeze(-1))
LL = torch.linalg.cholesky(KK + 1e-6 * torch.eye(len_observations).unsqueeze(0) )
noise = matmul(LL, torch.randn(dim_simulation, len_observations, num_observations)).permute(2, 1, 0).detach()
lorenz_nor = lorenz_nor + noise


# Unfold dynamics for ploting
lorenz_nor = lorenz_nor.reshape(num_simulation * len_observations, dim_simulation)
lorenz_nor = lorenz_nor[..., :dim_latent_true]

# True Latents
latent_true_unfolded = lorenz_nor
latent_true = latent_true_unfolded.reshape(num_observations, len_observations, dim_latent_true)

# Plot Lorenz Attractor
if dim_latent_true == 3:
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(lorenz_nor[:, 0], lorenz_nor[:, 1], lorenz_nor[:, 2], lw=0.5, color='k')
    ax.set_xlabel("Z[1]")
    ax.set_ylabel("Z[2]")
    ax.set_zlabel("Z[3]")
    ax.set_title("True Latent: Lorenz Attractor")
elif dim_latent_true == 2:
    plt.figure()
    plt.plot(lorenz_nor[:, 0], lorenz_nor[:, 1], lw=0.5, color='k')
    plt.xlabel("Z[1]")
    plt.ylabel("Z[2]")
    plt.title("True Latent: Lorenz Attractor")
else:
    plt.figure()
    yy = lorenz_nor.reshape(num_simulation, len_observations, dim_latent_true)
    for nn in range(num_observations):
        plt.plot(yy[nn], color='k')
    plt.title("True Latent: Lorenz Attractor")
    plt.ylabel('Z[1]')
    plt.xlabel('Time [a.u]')


#%%


# Map to observations
C = torch.rand(lorenz_nor.shape[-1], dim_observations)
C = C[:, C[0].sort(descending=True)[1]]

# Rates
rates_unfoled = matmul(lorenz_nor, C)
rates = rates_unfoled.reshape(num_observations, len_observations, dim_observations)

# Observations
#observations = rates + 0.3 * torch.randn(rates.shape)
observations = torch.poisson(10*torch.exp(rates - rates.min()))
observations_unfoled = observations.reshape(num_observations * len_observations, dim_observations)


# Plot Observation Summary
plt.figure(figsize=(3*4, 4))
plot_n = 2
plot_num = 10
plot_id = np.random.choice(dim_observations, plot_num)
cmap = 'inferno'

# Plot Observations
plt.subplot(2, 3, 1)
plt.imshow(observations[plot_n].transpose(-1, -2), aspect='auto', cmap=cmap)
plt.ylabel('Neurons')
plt.title('Observation#' + str(plot_n))

# Plot Rates
plt.subplot(2, 3, 4)
plt.imshow(rates[plot_n].transpose(-1, -2), aspect='auto', cmap=cmap)
plt.title('Rates')
plt.ylabel('Neurons')

# Plot Observations
plt.subplot(2, 3, 6)
plt.plot(observations[plot_n, :, plot_id].numpy())
plt.title('Observation#' + str(plot_n) + ' Neurons ')

# Plot Mapping
plt.subplot(2, 3, 3)
plt.imshow(C.transpose(-1, -2), aspect='auto', cmap='inferno')
plt.title('Mapping')

# Plot True Latent
plt.subplot(2, 3, 2)
plt.imshow(latent_true[plot_n].transpose(-1, -2), aspect='auto', cmap=cmap)
plt.ylabel('True')
plt.title('Latent#' + str(plot_n))

# Plot True Latents
plt.subplot(2, 3, 5)
plt.plot(latent_true[plot_n].numpy())
plt.tight_layout()
plt.title('Latent#' + str(plot_n))

#%%


# Normalise observations.
observations = observations.reshape(num_observations * len_observations, dim_observations)
o_mean, o_std = torch.mean(observations, dim=0, keepdim=True), torch.std(observations, dim=0, keepdim=True)
observations = (observations - o_mean) / o_std
observations = observations.reshape(num_observations, len_observations, dim_observations)

# Move to GPU if necessary
observations = torch.tensor(observations, dtype=dtype, device=device)
observation_locations = torch.linspace(0, 1, len_observations, dtype=dtype, device=device).unsqueeze(-1)
inducing_locations = torch.linspace(0, 1, num_inducing_points, dtype=dtype, device=device).unsqueeze(-1)

# Break Into Multiple Factors
num_factors = 3
num_full = (np.floor(dim_observations / num_factors)).astype(int)
obs = tuple([observations[..., num_full*i:num_full*(i+1)] for i in range(num_factors)])

# Linear / Non Linear Network
linear_networks = True
dim_hidden0 = [] if linear_networks else [10, 10]
non_lineraity0 = torch.nn.Identity() if linear_networks else F.relu

# Copy for each factors
dim_hidden = tuple([dim_hidden0 for _ in range(num_factors)])
neural_net = tuple(['perceptron' for _ in range(num_factors)])
nonlinearity = tuple([non_lineraity0 for _ in range(num_factors)])

#%%

from recognition_parametrised_model import RPM
import torch.nn.functional as F

observation_locations = torch.linspace(0, 1, len_observations).unsqueeze(-1)
inducing_locations = observation_locations[
    torch.floor(torch.linspace(0, len_observations-1, 50)).numpy().astype(int)
]

prior_params = {
    'gp_kernel': 'RBF',
    'optimizer': {'name': 'RMSprop', 'param': {'lr': 1e-3}},
    'scale': 1,
    'lengthscale': 0.01,
}

factors_params = {
    'channels': [[], [], []],
    'kernel_conv': [[], [], []],
    'kernel_pool': [[], [], []],
    'dim_hidden': [[10, 10], [10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu, F.relu],
    'covariance': ['fixed', 'fixed', 'fixed'],
    'optimizer': {'name': 'RMSprop', 'param': {'lr': 1e-2}},
}

auxiliary_params = {
    'channels': [[], [], []],
    'kernel_conv': [[], [], []],
    'kernel_pool': [[], [], []],
    'dim_hidden': [[10, 10], [10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu, F.relu],
    'covariance': ['fixed', 'fixed', 'fixed'],
    'optimizer': {'name': 'RMSprop', 'param': {'lr': 1e-2}},
}

variational_params = {
    'inference_mode': 'parametrized',  # 'amortized', 'parametrized'
    'channels': [[], [], []],
    'kernel_conv': [[], [], []],
    'kernel_pool': [[], [], []],
    'dim_hidden': [[10, 10], [10, 10], [10, 10]],
    'dim_hidden_merged': [],
    'non_linearity': [F.relu, F.relu, F.relu],
    'non_linearity_merged': F.relu,
    'covariance': 'full',
    'optimizer': {'name': 'RMSprop', 'param': {'lr': 1e-2}},
}

fit_params = {
    'num_epoch': 10,
    'dim_latent': 3,
    'prior_params': prior_params,
    'factors_params': factors_params,
    'auxiliary_params': auxiliary_params,
    'variational_params': variational_params,
}

rpm = RPM(
    observations=obs,
    observation_locations=observation_locations,
    inducing_locations=inducing_locations,
    fit_params=fit_params,
)

rpm.fit(obs)

#%%


# from save_load import rpm_load, rpm_save
# rpm_save(rpm, './tmp.pickle')
# rpm_loaded = rpm_load('./tmp.pickle', observations=observations)

#%%



#%%




#%%

from utils_process import plot_rpgpfa_summary

plot_rpgpfa_summary(
    rpm=rpm,
    plot_id_factors=[0],
    plot_id_observations=[1],
    plot_variational=True,
    plot_regressed=True,
    plot_variance=True,
    plot_true=True,
    latent_true=latent_true,
    regress_param=None,
    plot_type='linear',
)


#%%

# plot_factors_prior(
#     model,
#     tt_index=0,
#     factor_id=0,
#     num_std=5,
#     num_landscape=50
# )

from flexible_multivariate_normal import FlexibleMultivariateNormal
from utils import diagonalize



from utils_process import plot_rpgpfa_mixture

plot_rpgpfa_mixture(
    rpm,
    plot_id_factor=0,
    plot_id_index=0,
    plot_locations=50,
    plot_num_std=5,
)





