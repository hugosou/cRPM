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
import imageio
import pickle
# %%


# GPUs ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data type: float64 / float32
data_type = torch.float32
torch.set_default_dtype(data_type)

# Stored Videos
# trajectory_folder = '../videos_rpm/'
# trajectory_name = 'trajectories_param.pkl'

trajectory_folder = '/Users/hugosoulat/Documents/PYTHON/videos_rpm/'
trajectory_name = 'trajectories_param.pkl'


#%%

# Load Trajectories
with open(trajectory_folder + trajectory_name, "rb") as input_file:
        trajectories = pickle.load(input_file)


# ~ N x Agent x T x 2
spatial_trajectories = trajectories['spatial_trajectories']

# ~ N x T x 4
distance_from_fixed = trajectories['distance_from_fixed']

# Id.
main_agent = trajectories['main_agent']

# N x T x W x H
video_tensor = trajectories['video_tensor']
num_observations_full, len_observation, width, height = video_tensor.shape

# ~ N x T x 2
main_trajectory = spatial_trajectories[:, main_agent]


#%% Plots

# Plot videos and sound recordings for one observation (if not on GPU)
plot_observations = not(torch.cuda.is_available())

if plot_observations:

    # Microphone Locations
    mic_position = torch.tensor([[-1, -1], [-1, +1], [+1, -1], [+1, +1]], device=device, dtype=data_type)
    mic_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # 'Color' of the agents
    c1 = [0.8, 0.8, 0.8]
    c2 = [0.5, 0.5, 0.5]
    c3 = [0.3, 0.3, 0.3]
    color_agents = [c1, c2, c3]

    obs_eg = 6
    tplot = [0, 100, 150]

    pheigh = int(np.floor(np.sqrt(len(tplot)+2)))
    pwidth = int(np.ceil((len(tplot)+2) / pheigh))

    pheigh = 1
    pwidth = int(np.ceil((len(tplot) + 2) / pheigh))

    plt.figure(figsize=(pwidth * 4, pheigh * 4))

    for tt_id in range(len(tplot)):
        name = 'video_' + 'full_plot_n' + str(obs_eg).zfill(3) + '_t' + str(tplot[tt_id]).zfill(4) + '.png'
        im = imageio.imread(trajectory_folder + name)
        print(name)
        plt.subplot(pheigh, pwidth, tt_id + 1)
        plt.imshow(im)
        plt.imshow(video_tensor[obs_eg, tplot[tt_id]], cmap='gray')

        plt.title('t = ' + str(tplot[tt_id]) + '/' + str(len_observation))
        plt.xlim([75, 550])
        plt.ylim([550, 75])
        plt.axis('off')

    plt.subplot(pheigh, pwidth, tt_id + 2)
    for ii in range(mic_position.shape[0]):
        plt.scatter(mic_position[ii, 0], mic_position[ii, 1], s=100, c=mic_colors[ii], marker='s')
    plt.plot(main_trajectory[obs_eg, :, 0], main_trajectory[obs_eg, :, 1], c=color_agents[main_agent])
    plt.scatter(main_trajectory[obs_eg, 0, 0], main_trajectory[obs_eg, 0, 1], s=100, c=color_agents[main_agent],
                label='Start')
    plt.title('Top View')
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlabel('z1[t]')
    plt.ylabel('z2[t]')
    plt.legend()

    plt.subplot(pheigh, pwidth, tt_id + 3)
    for ii in range(mic_position.shape[0]):
        plt.plot(np.linspace(0, 1, len_observation), distance_from_fixed[obs_eg, :, ii], c=mic_colors[ii],
                    label='mic.' + str(ii))
    plt.legend(loc=5)
    plt.title('Distance(mic.)')
    plt.xlabel('Time [a.u]')


#%%

#%% Fit
num_observation = 10

# Build Observed Factors (Transfer to GPU if necessary)
observations1 = torch.tensor(
    video_tensor[:num_observation, :, 1:][..., 1:-2],
    device=device,
    dtype=data_type
)
observations1 = (1 - observations1 / observations1.max())
observations2 = torch.tensor(distance_from_fixed[:num_observation], device=device, dtype=data_type)


def normalize_observations(obs, num_event_dim=1):

    full_dim = obs.shape
    batch_dim = torch.tensor(obs.shape[:num_event_dim])
    event_dim = torch.tensor(obs.shape[num_event_dim:])

    obs = obs.reshape(batch_dim.prod(), event_dim.prod())
    o_mean, o_std = torch.mean(obs, dim=0, keepdim=True), torch.std(obs, dim=0, keepdim=True)
    obs = (obs - o_mean) / (o_std + 1e-6)
    obs = obs.reshape(full_dim)

    return obs

observations1 = normalize_observations(observations1, num_event_dim=2)
observations2 = normalize_observations(observations2, num_event_dim=2)

observations = [observations1, observations2]


use_sound_speech = False
len_snippet = 100
dim_latent = 2
num_inducing = 50

observation_locations = torch.linspace(0, 1, len_observation, device=device, dtype=data_type).unsqueeze(-1)
inducing_locations = observation_locations[np.floor(np.linspace(0, len_observation-1, num_inducing).astype(int))]

#%%

from recognition_parametrised_model import RPM
import torch.nn.functional as F

prior_params = {
    'gp_kernel': 'RBF',
    'scale': 1,
    'lengthscale': 0.01,
    'fit_kernel_scale': False,
    'fit_kernel_scale_prior': False,
    'fit_kernel_lengthscale': True,
    'fit_kernel_lengthscale_prior': False,
    'fit_prior_mean_param': False,
    'optimizer': lambda params: torch.optim.Adam(params=params, lr=1e-3),
}

factors_params = {
    'channels': [[1, 10, 10], []],
    'kernel_conv': [[2, 2], []],
    'kernel_pool': [[2, 2], [], []],
    'dim_hidden': [[10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu],
    'covariance': ['fixed', 'fixed'],
    'optimizer': lambda params: torch.optim.Adam(params=params, lr=1e-3),
}

auxiliary_params = {
    'channels': [[1, 10, 10], []],
    'kernel_conv': [[2, 2], []],
    'kernel_pool': [[2, 2], [], []],
    'dim_hidden': [[10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu],
    'covariance': ['fixed', 'fixed'],
    'optimizer': lambda params: torch.optim.Adam(params=params, lr=1e-3),
}

variational_params = {
    'inference_mode': 'parametrized',  # 'amortized', 'parametrized'
    'channels': [[1, 10, 10], []],
    'kernel_conv': [[2, 2], []],
    'kernel_pool': [[2, 2], [], []],
    'dim_hidden': [[10, 10], [10, 10]],
    'dim_hidden_merged': [],
    'non_linearity': [F.relu, F.relu, F.relu],
    'non_linearity_merged': F.relu,
    'covariance': 'full',
    'optimizer': lambda params: torch.optim.Adam(params=params, lr=1e-3),
}

fit_params = {
    'num_epoch': 2000,
    'batch_size': num_observation,
    'dim_latent': 2,
    'prior_params': prior_params,
    'factors_params': factors_params,
    'auxiliary_params': auxiliary_params,
    'variational_params': variational_params,
    'ergodic': False,
}

rpm = RPM(
    observations=observations,
    observation_locations=observation_locations,
    inducing_locations=inducing_locations,
    fit_params=fit_params,
)

rpm.fit(observations)



#%%

from importlib import reload
from utils_process import plot_rpgpfa_summary, plot_loss
import utils_process
utils_process = reload(utils_process)


plot_loss(rpm, offset=0)


#%%
latent_true = main_trajectory[:num_observation]

utils_process.plot_rpgpfa_summary(
    rpm=rpm,
    plot_id_factors=[1],
    plot_id_observations=[3],
    plot_variational=False,
    plot_regressed=True,
    plot_variance=False,
    plot_true=True,
    latent_true=latent_true,
    regress_param=None,
    plot_type='linear',
)



