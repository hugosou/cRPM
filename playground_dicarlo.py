

import brainscore_vision
import torch
import numpy as np

# Imports
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from recognition_parametrised_model import RPM

from matplotlib import pyplot as plt
from save_load import load_crpm, save_crpm

import torch
import torch.nn.functional as F



neural_data = brainscore_vision.load_dataset("MajajHong2015.public")
#neural_data = neural_data.transpose('presentation', 'neuroid', 'time_bin')
#neural_data



# Compact data
compact_data = neural_data.multi_groupby(['category_name', 'object_name', 'stimulus_id', 'variation']).mean(dim='presentation')  # (1)
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

# Find trials with given level of difficulty
#select_variation = np.where(variation >= 0)[0]
select_variation = np.where(variation == 3)[0]
select_variation = np.where(variation == 0)[0]

# Filter the dataset Using the trials with given level of difficulty
filtered_neural = np.array(neural_observation[select_variation])
filtered_latent_basic = basic_index[select_variation]
filtered_latent_categories = categories_index[select_variation]

#%%





#%%

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
    path_cur = compact_data['stimulus_id'].data[select_variation][ii]
    path_cur = stimulus_set.get_stimulus(path_cur)
    ima_cur = image.imread(path_cur)[:, :, 0]
    filtered_images[ii] = torch.tensor(ima_cur)

# Downsample images
ds_factor = 0.5
filtered_images = torch.nn.functional.interpolate(filtered_images.unsqueeze(dim=1), scale_factor=ds_factor, mode='bilinear')
filtered_images = filtered_images.squeeze(dim=1)
filtered_images = filtered_images[:, 20:110][:, :, 20:110]
#%%


#%%

#%% Plot
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


#%%

#%% Builds

# Build Observations
observations1 = filtered_images
observations2 = torch.tensor(filtered_neural)

# Random train /test split
train_test_perm = torch.randperm(num_observations_tot)
train_length = int(num_observations_tot/2)
train_indices = train_test_perm[:train_length]
tests_indices = train_test_perm[train_length:]

observations1_train = observations1[train_indices]
observations2_train = observations2[train_indices]
observations1_tests = observations1[tests_indices]
observations2_tests = observations2[tests_indices]

true_latent_train = filtered_latent_basic[train_indices]
true_latent_tests = filtered_latent_basic[tests_indices]

# GPUs ?
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Gather and move obs
observations_train = (observations1_train.to(device).unsqueeze(1), observations2_train.to(device).unsqueeze(1))
observations_tests = (observations1_tests.to(device).unsqueeze(1), observations2_tests.to(device).unsqueeze(1))


#%%


from recognition_parametrised_model import RPM
import torch.nn.functional as F

dim_latent = 2


prior_params = {
    'gp_kernel': 'RBF',
    'optimizer': {'name': 'Adam', 'param': {'lr': 1e-3}}
}

factors_params = {
    'channels': [[1, 20, 20], []],
    'kernel_conv': [[2, 2], []],
    'kernel_pool': [[1, 1], []],
    'dim_hidden': [[10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu],
    'covariance': ['fixed', 'fixed'],
    'optimizer': {'name': 'Adam', 'param': {'lr': 1e-3}}
}

auxiliary_params = {
    'channels': [[1, 20, 20], []],
    'kernel_conv': [[2, 2], []],
    'kernel_pool': [[1, 1], []],
    'dim_hidden': [[10, 10], [10, 10]],
    'non_linearity': [F.relu, F.relu],
    'covariance': ['full', 'full'],
    'optimizer': {'name': 'Adam', 'param': {'lr': 1e-3}},
}

variational_params = {
    'inference_mode': 'amortized',  # 'amortized', 'parametrized'
    'channels': [[1, 20, 20], []],
    'kernel_conv': [[2, 2], []],
    'kernel_pool': [[1, 1], []],
    'dim_hidden': [[10, 10], [10, 10]],
    'dim_hidden_merged': [10, 10],
    'non_linearity': [F.relu, F.relu],
    'non_linearity_merged': F.relu,
    'covariance': 'diag',
    'optimizer': {'name': 'Adam', 'param': {'lr': 1e-3}}
}


fit_params = {
    'num_epoch': 200,
    'dim_latent': dim_latent,
    'prior_params': prior_params,
    'factors_params': factors_params,
    'auxiliary_params': auxiliary_params,
    'variational_params': variational_params,
}

len_observations = 1
observation_locations = torch.linspace(0, 1, len_observations).unsqueeze(-1)

rpm = RPM(
    observations=observations_train,
    observation_locations=observation_locations,
    fit_params=fit_params,
)

rpm.fit(observations_train)

print(0)

#%%

from matplotlib import pyplot as plt

plt.figure()
plt.plot(rpm.loss_tot[3:])
plt.show()


print(0)



#%%


# cur_eval = 'train'
#
# if cur_eval == 'test':
#     observations_cur = observations_tests
#     latent_true = true_latent_tests
#
# elif cur_eval == 'train':
#     observations_cur = observations_train
#     latent_true = true_latent_train
#
# with torch.no_grad():
#     rpm._update_factors(observations_cur)
#     rpm._update_variational(observations_cur)




#%%

posterior_mean = rpm.dist_marginals.suff_stat_mean[0].squeeze(1).detach()
latent_true = true_latent_train

plot_width = dim_latent - 1
plot_heigh = dim_latent - 1
plt.figure(figsize=(5 * plot_width, 5 * plot_heigh))
for ii in np.arange(dim_latent):
    for jj in np.arange(ii + 1, dim_latent):
        plt.subplot(plot_heigh, plot_width, jj + ii * (dim_latent - 1))
        for class_id in range(len(np.unique(latent_true))):
            mask = latent_true == class_id
            plt.scatter(posterior_mean[mask, jj], posterior_mean[mask, ii], label=label_basic[class_id])
            plt.xlabel('Dim ' + str(jj))
            plt.ylabel('Dim ' + str(ii))

        if ii == 0 and jj == 1:
            plt.legend()





#%%