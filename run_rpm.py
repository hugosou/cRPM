# Imports
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from recognition_parametrised_model import RPM

from matplotlib import pyplot as plt
from save_load import load_crpm, save_crpm

import torch
import torch.nn.functional as F

import brainscore

from utils import plot_confusion, get_color


# %%

# Load brainscore data
neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")

# Compact data
compact_data = neural_data.multi_groupby(
    ['category_name', 'object_name', 'stimulus_id', 'variation', 'rxy', 'rxz', 'ryz', 'ty', 'tz', 'rxy_semantic',
     'ryz_semantic', 's', 'background_id']).mean(dim='presentation')  # (1)
compact_data = compact_data.sel(region='IT')  # (2)
compact_data = compact_data.squeeze('time_bin')  # (3)
compact_data = compact_data.transpose('presentation', 'neuroid')  # (4)

# All Images
stimulus_set = neural_data.attrs['stimulus_set']

# Neural Recordings
neural_observation = compact_data.data

# Stimulus labels
label_basic, basic_index = np.unique(compact_data['category_name'].data, return_inverse=True)
label_categories, categories_index = np.unique(compact_data['object_name'].data, return_inverse=True)

# Level of difficulty/variation for each images
variation = compact_data['variation'].data
# select_variation = variation >= 0
# select_variation = variation == 3
select_variation = variation == 0

# Basic Categories Kept
categories = 'all'  # basic, all, Faces, Animals, ...
if categories == 'all' or categories == 'basic':
    select_categories = basic_index >= 0
else:
    select_categories = label_basic[basic_index] == categories
select_trial = select_variation * select_categories

# Filter the dataset Using the trials with given level of difficulty and corresponding Categories
filtered_neural = neural_observation[select_trial]
filtered_latent_basic = basic_index[select_trial]
filtered_latent_categories = categories_index[select_trial]

rx = torch.tensor(compact_data['ryz'].data[select_trial]).unsqueeze(dim=1)
ry = torch.tensor(compact_data['rxz'].data[select_trial]).unsqueeze(dim=1)
rz = torch.tensor(compact_data['rxy'].data[select_trial]).unsqueeze(dim=1)
filtered_rotation = torch.cat((rx, ry, rz), dim=1)

filtered_size = torch.tensor(compact_data['s'].data[select_trial]).unsqueeze(dim=1)

ty = torch.tensor(compact_data['ty'].data[select_trial]).unsqueeze(dim=1)
tz = torch.tensor(compact_data['tz'].data[select_trial]).unsqueeze(dim=1)
filtered_translation = torch.cat((ty, tz), dim=1)

if categories == 'basic':
    filtered_labels = label_basic
    filtered_latent = filtered_latent_basic
else:
    filtered_labels = label_categories
    filtered_latent = filtered_latent_categories

# Colors for plotting
cmap0 = plt.cm.tab10(np.linspace(0, 1, 10))
colors_basic, colors_categories = get_color(basic_index, index_sub=categories_index, cmap=cmap0, amp=3)

#%%  Downsample and Crop Images

# Combined Train and Test set sizes
num_observations_tot = len(filtered_neural)
assert num_observations_tot == len(filtered_latent_basic)
assert num_observations_tot == len(filtered_latent_categories)

# Gather images
images_height = 256
images_width = 256

# Gather Images
filtered_images = torch.zeros(num_observations_tot, images_height, images_width)
for ii in range(num_observations_tot):
    path_cur = compact_data['stimulus_id'].data[select_trial][ii]
    path_cur = stimulus_set.get_stimulus(path_cur)
    ima_cur = image.imread(path_cur)[:, :, 0]
    filtered_images[ii] = torch.tensor(ima_cur)

# Downsample images
ds_factor = 0.50
filtered_images = torch.nn.functional.interpolate(filtered_images.unsqueeze(dim=1), scale_factor=ds_factor,
                                                  mode='bilinear')
filtered_images = filtered_images.squeeze(dim=1)

# Crop Images
centerw = int(np.round(filtered_images.shape[-2] / 2))
centerh = int(np.round(filtered_images.shape[-1] / 2))

radiusw = int(filtered_images.shape[-2] - centerw)
radiush = int(filtered_images.shape[-1] - centerh)
crop_pct = 0.25
crop_pct = 0.5

crop_radiusw = int(radiusw * (1 - crop_pct))
crop_radiusw = int(radiush * (1 - crop_pct))

filtered_images = filtered_images[:, centerw - crop_radiusw: centerw + crop_radiusw]
filtered_images = filtered_images[:, :, centerh - crop_radiusw: centerh + crop_radiusw]

# %% Plot

do_plot = True
if do_plot:
    num_plot = 35
    plot_check = np.arange(0, num_observations_tot, int(num_observations_tot / num_plot))

    figh = int(np.ceil(np.sqrt(len(plot_check))))
    figw = int(np.ceil(np.sqrt(len(plot_check))))

    plt.figure(figsize=(3 * figw, 3 * figh))
    for ii, id in enumerate(plot_check):
        plt.subplot(figh, figw, ii + 1)
        lab_basic = label_basic[filtered_latent_basic[id]]
        lab_categories = label_categories[filtered_latent_categories[id]]

        plt.imshow(filtered_images[id], cmap='gray', aspect='auto')
        plt.title(lab_basic + ' : ' + lab_categories)
        plt.xticks([])
        plt.yticks([])

        if ii == 0:
            plt.ylabel('Downsampling Factor ' + str(ds_factor))

# %% Builds

# Build Observations
observations1 = filtered_images
observations2 = torch.tensor(filtered_neural)

# Random train /test split
train_test_perm = torch.randperm(num_observations_tot)
train_length = int(3 * num_observations_tot / 4)
train_indices = train_test_perm[:train_length]
tests_indices = train_test_perm[train_length:]

observations1_train = observations1[train_indices]
observations2_train = observations2[train_indices]
observations1_tests = observations1[tests_indices]
observations2_tests = observations2[tests_indices]

true_latent_train = filtered_latent[train_indices]
true_latent_tests = filtered_latent[tests_indices]

# GPUs ?
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Gather and move obs
observations_train = (observations1_train.to(device), observations2_train.to(device))
observations_tests = (observations1_tests.to(device), observations2_tests.to(device))

rotation_train = filtered_rotation[train_indices]
rotation_tests = filtered_rotation[tests_indices]

translation_train = filtered_translation[train_indices]
translation_tests = filtered_translation[tests_indices]

rotation_train = filtered_rotation[train_indices]
rotation_tests = filtered_rotation[tests_indices]

translation_train = filtered_translation[train_indices]
translation_tests = filtered_translation[tests_indices]

#%% Shuffle pairs and use multi/uni-modal input
do_shuffle = True

if do_shuffle:

    # Select What Type of factors to use [0: images, 1: neural]
    factors = [0, 1]

    # Number of pairs
    num_observations_shuffled = 5000

    # All indices of the current dataset
    all_classes = np.unique(true_latent_train)
    num_classes = len(all_classes)

    # Random list of true latent indices
    shuffled_classes = all_classes[
        torch.round((num_classes - 1) * torch.rand(num_observations_shuffled)).numpy().astype(int)]

    # Init Training Observations
    observations_shuffled = [torch.zeros(num_observations_shuffled, *observations_train[ii].shape[1:]) for ii in
                             factors]

    # Randomly assign each observation
    for ii in range(num_observations_shuffled):

        # Sample observations
        mask = np.where(shuffled_classes[ii] == true_latent_train)[0]
        observation_indices = torch.randperm(len(mask))[:len(factors)]

        # Assign it to each factors
        for jj in range(len(factors)):
            observations_shuffled[jj][ii] = observations_train[factors[jj]][mask[observation_indices[jj]]]

    # Move and Copy (if necessary, for the test set) observaions
    observations_train = [ii.to(device) for ii in observations_shuffled]
    observations_tests = [observations_tests[ii].to(device) for ii in factors]

    true_latent_train = torch.tensor(shuffled_classes)
    true_latent_tests = torch.tensor(true_latent_tests)

observations_train = [ii.unsqueeze(1) if len(ii.shape) == 3 else ii for ii in observations_train]
observations_tests = [ii.unsqueeze(1) if len(ii.shape) == 3 else ii for ii in observations_tests]


#%%

dim_latent = 3
dim_inputs = [observations_train[0].shape[-2:], observations_train[1].shape[-1:]]
kernel_convs = [[3, 3, 3, 3], []]
kernel_pools = [[2, 2, 2, 2], []]
channelss = [[1, 20, 20, 10, 10], []]
dim_hiddens = [[30, 30], [130]]
dim_hidden_merged = [20]

# kernel_convs = [[], []]
# kernel_pools = [[], []]
# channelss = [[], []]
# dim_hiddens = [[20, 20], [20, 20]]
# dim_hidden_merged = [20]

factors_params = {'channels': channelss,
                  'kernel_conv': kernel_convs,
                  'kernel_pool': kernel_pools,
                  'dim_hidden': dim_hiddens,
                  'covariance': 'fixed',
                  'nonlinearity': F.relu}

variational_params = {'channels': channelss,
                      'kernel_conv': kernel_convs,
                      'kernel_pool': kernel_pools,
                      'dim_hidden': dim_hiddens,
                      'dim_hidden_merged': dim_hidden_merged,
                      'covariance': 'fixed_diag',
                      'nonlinearity': F.relu}

# Fit parameters
fit_params = {'num_epoch': 50000,
              'optimizer_factors': {'name': 'Adam', 'param': {'lr': 0.5e-3, 'weight_decay': 0.0001}},
              'optimizer_variational': {'name': 'Adam', 'param': {'lr': 0.5e-3, 'weight_decay': 0.0001}},
              'factors_params': factors_params,
              'variational_params': variational_params,
              'pct': 0.01,
              'minibatch_size': 1000,
              }

model = RPM(dim_latent, observations_train, fit_params=fit_params)

model.fit(observations_train)

save_crpm('./batch_test.pkl', model, observations=observations_train)

#%%

plt.figure()
plt.plot(model.loss_tot, c='k')
plt.xlabel('epoch')
plt.ylabel('- Free Energy')