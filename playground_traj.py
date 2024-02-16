import torch
import numpy as np
from torch import matmul
import matplotlib.pyplot as plt

from utils_demo import generate_lorenz
from kernels import RBFKernel

import torch
import numpy as np
from torch import matmul
import matplotlib.pyplot as plt
from kernels import RBFKernel
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import imageio
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

trajectory_folder = '/nfs/ghome/live/hugos/Documents/PYTHON/videos_rpm/'
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
num_observation = 50
# Build Observed Factors (Transfer to GPU if necessary)
observations1 = torch.tensor(
    video_tensor[:num_observation, :, 1:][..., 1:-2],
    device=device,
    dtype=data_type
)
observations1 = (1 - observations1 / observations1.max())
observations2 = torch.tensor(distance_from_fixed[:num_observation], device=device, dtype=data_type)

#%%

def get_speech_samples(distance_from_fixed, audio_path, downsample=10, len_snippet=1000, normalize=True):

    num_observations, len_observation, _ = distance_from_fixed.shape
    data_type = distance_from_fixed.dtype
    device = distance_from_fixed.device

    speech_snips = np.zeros((num_observations, len_observation, len_snippet))

    cur_obs = 0
    cur_time = 0

    # Grasp all audio files
    for path in Path(audio_path).rglob('*.wav'):

        # Grasp data
        data, samplerate = sf.read(path)

        # Downsample data
        data = data[::downsample]

        # Get the length (in second) of each audio snippet
        samplerate = samplerate / downsample
        time_snippet_sec = len_snippet / samplerate


        # Number of snippet in the current data file
        snippet_num_cur = int(np.floor(len(data)/len_snippet))

        for inds in range(snippet_num_cur):

            # Break if filled all observations with sound snippet
            if cur_obs >= num_observations:
                break

            # Fill Current observation and current time with sound snippet
            speech_snips[cur_obs, cur_time] = data[len_snippet * inds:len_snippet * (inds + 1)]

            # Update Time
            cur_time += 1

            # If all time point have been filled, move to next observation
            if cur_time >= len_observation:
                cur_obs += 1
                cur_time = 0

    # Error if you didn't fill the array
    if cur_obs < num_observations:
        raise Exception('Not enough speech snippets in timit')

    # Get speech samples
    speech_data = torch.tensor(speech_snips, dtype=data_type, device=device)
    if normalize:
        speech_data = speech_data - speech_data.mean(dim=-1, keepdim=True)
        speech_data = speech_data / speech_data.max(dim=-1)[0].unsqueeze(-1)

    # Maximum distance in the square arena
    dmax = torch.sqrt(torch.tensor([8]))

    # Enveloppe signal
    modulation = (dmax - distance_from_fixed) / dmax

    # Modulate
    distance_modulated_speech = (modulation.unsqueeze(-1) * speech_data.unsqueeze(-2))

    return distance_modulated_speech, time_snippet_sec

#%%

from pathlib import Path
import soundfile as sf
len_snippet = 100
audio_path = '/Users/hugosoulat/Documents/PYTHON/timit/train/'
audio_path = '/nfs/ghome/live/hugos/Documents/PYTHON/timit/train/'
distance_modulated_audio, time_snippet_sec = get_speech_samples(
    distance_from_fixed,
    audio_path,
    downsample=20,
    len_snippet=len_snippet,
    normalize=True
)
#%%







