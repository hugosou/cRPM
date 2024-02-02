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
from torch import nn

from utils import plot_confusion, get_color

#%%

# GPUs ?
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
select_variation = variation == 3
select_variation = variation == 0

# Basic Categories Kept
categories = 'basic'  # basic, all, Faces, Animals, ...
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



# %% Downsample and Crop Images

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
ds_factor = 0.75
filtered_images = torch.nn.functional.interpolate(filtered_images.unsqueeze(dim=1), scale_factor=ds_factor,
                                                  mode='bilinear')
filtered_images = filtered_images.squeeze(dim=1)

# Crop Images
centerw = int(np.round(filtered_images.shape[-2] / 2))
centerh = int(np.round(filtered_images.shape[-1] / 2))

radiusw = int(filtered_images.shape[-2] - centerw)
radiush = int(filtered_images.shape[-1] - centerh)
crop_pct = 0.2
crop_pct = 0.33
# crop_pct = 0

crop_radiusw = int(radiusw * (1 - crop_pct))
crop_radiusw = int(radiush * (1 - crop_pct))

filtered_images = filtered_images[:, centerw - crop_radiusw: centerw + crop_radiusw]
filtered_images = filtered_images[:, :, centerh - crop_radiusw: centerh + crop_radiusw]


# %% Plot
# Check examples

do_plot = True
if do_plot:
    num_plot = 60
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
train_length = int(0.8 * num_observations_tot)
train_indices = train_test_perm[:train_length]
tests_indices = train_test_perm[train_length:]

observations1_train = observations1[train_indices]
observations2_train = observations2[train_indices]
observations1_tests = observations1[tests_indices]
observations2_tests = observations2[tests_indices]

true_latent_train = filtered_latent[train_indices]
true_latent_tests = filtered_latent[tests_indices]

true_category_train = filtered_latent_basic[train_indices]
true_category_tests = filtered_latent_basic[tests_indices]

true_identity_train = filtered_latent_categories[train_indices]
true_identity_tests = filtered_latent_categories[tests_indices]

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

#%% Create Pseudo Observations


# Shuffle pairs and use multi/uni-modal input
do_shuffle = False

if do_shuffle:

    # Select What Type of factors to use [0: images, 1: neural]
    factors = [0, 1]

    # Number of pairs
    num_observations_shuffled = 9000

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


# %%
from save_load import load_crpm
do_load = True

if do_load:
    name = './low_variation_crpm.pkl'
    model_loaded, observations, true_latent = load_crpm(name)
    model = model_loaded
    model.recognition_variational.to(device)
    [i.to(device) for i in model.recognition_factors]
    dim_latent = model.dim_latent


#%%  Plot Loss

model.recognition_variational.to(device)
[i.to(device) for i in model.recognition_factors]
print('Moved To GPU')

[ii.eval() for ii in model.recognition_factors]
model.recognition_variational.eval()
print('Eval Mode')

plt.figure()
plt.plot(model.loss_tot, c='k')
plt.xlabel('epoch')
plt.ylabel('- Free Energy')


#%% # Display Latent MAP

cur_eval = 'train'

if cur_eval == 'test':
    observations_cur = observations_tests
    latent_true = true_latent_tests

elif cur_eval == 'train':
    observations_cur = observations_train
    latent_true = true_latent_train

with torch.no_grad():
    # observations_cur = [i.to("cpu") for i in observations_cur]
    # latent_true = torch.tensor(latent_true)
    model._update_factors(observations_cur)
    model._update_variational(observations_cur)

    epoch_batch = model.epoch_batch
    # epoch_batch = [0, 0]
    batch_id = model.mini_batches[epoch_batch[0]][epoch_batch[1]]
    latent_true = torch.tensor(latent_true)[batch_id]

# J = 1
# posterior_mean = model.factors.suff_stat_mean[0][J].detach().cpu().numpy()
posterior_mean = model.variational.suff_stat_mean[0].detach().cpu().numpy()

translation_color = translation_train - translation_train.mean()
translation_color = translation_color / translation_color.abs().max()
translation_color = 0.5 * (translation_color + 1)
translation_color = torch.cat((translation_color, 0.5 * torch.ones(translation_color.shape[0], 1)), dim=-1)

rotation_color = rotation_train - rotation_train.mean()
rotation_color = rotation_color / rotation_color.abs().max()
rotation_color = 0.5 * (rotation_color + 1)

colors = colors_basic[latent_true]
# colors = colors_categories[latent_true]
# colors = colors_categories[true_identity_train]


# colors = translation_color[batch_id]
# colors = rotation_color

plot_width = dim_latent - 1
plot_heigh = dim_latent - 1
plt.figure(figsize=(5 * plot_width, 5 * plot_heigh))
for ii in np.arange(dim_latent):
    # for ii in np.arange(1):
    for jj in np.arange(ii + 1, dim_latent):
        plt.subplot(plot_heigh, plot_width, jj + ii * plot_width)
        plt.scatter(posterior_mean[:, jj], posterior_mean[:, ii], c=colors)
        plt.xlabel('Dim ' + str(jj))
        plt.ylabel('Dim ' + str(ii))

for class_id in range(len(colors_basic)):
    plt.scatter(0, 0, color=colors_basic[class_id], label=label_basic[class_id])
plt.legend()
plt.show()

# %%

from sklearn.cluster import KMeans

X = posterior_mean.astype('float64')

num_centroid = 5
num_neighboo = 5
kmeans = KMeans(n_clusters=num_centroid, random_state=0).fit(X)

centroids = 1.2 * kmeans.cluster_centers_
centroids_radius = np.argsort(-(centroids ** 2).sum(axis=-1))
centroids = centroids[centroids_radius]

# %%

# plot_width = dim_latent - 1
# plot_heigh = dim_latent - 1
# plt.figure(figsize =(5 * plot_width, 5 * plot_heigh))
# for ii in np.arange(dim_latent):
#     for jj in np.arange(ii + 1, dim_latent):
#         plt.subplot(plot_heigh, plot_width, jj + ii * (dim_latent-1))
#         plt.scatter(posterior_mean[:, jj], posterior_mean[:, ii], c=colors)
#         plt.scatter(centroids[:, jj], centroids[:, ii], c='k', s=300)
#         plt.xlabel('Dim ' + str(jj))
#         plt.ylabel('Dim ' + str(ii))

# %%

plt.figure(figsize=(3 * (num_neighboo + 1), 3 * num_centroid))

for jj in range(num_centroid):
    centroid = centroids[jj]
    dists = ((torch.tensor(posterior_mean) - centroid) ** 2).sum(dim=-1)
    _, indices = dists.sort()
    offset = 10
    indices = indices[offset:(offset + num_neighboo)]

    for ii in range(num_neighboo):

        id = indices[ii]
        id = batch_id[id]
        plt.subplot(num_centroid, len(indices) + 1, ii + 1 + jj * (num_neighboo + 1))
        # lab_basic = label_basic[latent_true[id]]

        plt.imshow(observations_cur[0][id, 0].to("cpu"), cmap='gray', aspect='auto')
        # plt.title(lab_basic)
        plt.xticks([])
        plt.yticks([])

        if ii == 0:
            plt.ylabel('Centroid ' + str(jj))

    plt.subplot(num_centroid, len(indices) + 1, ii + 2 + jj * (num_neighboo + 1))
    plt.scatter(posterior_mean[:, -1], posterior_mean[:, -2], c=colors)
    plt.scatter(centroids[jj, -1], centroids[jj, -2], c='k', s=300)

# %% md

# Train an SVM on the trained latent space. Test it on unseen data

# %%

from sklearn import svm

cur_eval = 'train'
if cur_eval == 'test':

    observations_cur = observations_tests
    latent_true = torch.tensor(true_latent_tests).to("cpu")

    # observations_cur = [observations_cur[0][:N], observations_cur[1][:N]]
    # latent_true = latent_true[:N]

    with torch.no_grad():
        model._update_factors(observations_cur, full=True)
        model._update_variational(observations_cur, full=True)


elif cur_eval == 'train':

    observations_cur = observations_train
    latent_true = torch.tensor(true_latent_train).to("cpu")

    # observations_cur = [observations_cur[0][:N], observations_cur[1][:N]]
    # latent_true = latent_true[:N]

    with torch.no_grad():
        model._update_factors(observations_cur)
        model._update_variational(observations_cur)

    epoch_batch = model.epoch_batch
    batch_id = model.mini_batches[epoch_batch[0]][epoch_batch[1]]
    latent_true = torch.tensor(latent_true)[batch_id]

posterior_mean = model.variational.suff_stat_mean[0].detach().cpu().numpy()

# Train SVM
svm_map, latent_true_svm = np.unique(latent_true, return_inverse=True)
# clf = svm.SVC(C=10,kernel='linear')
clf = svm.LinearSVC(C=50, max_iter=100000, tol=0.000001)

clf.fit(posterior_mean, latent_true_svm)

cur_eval = 'test'
if cur_eval == 'test':

    observations_cur = observations_tests
    latent_true = torch.tensor(true_latent_tests).to("cpu")

    # observations_cur = [observations_cur[0][:N], observations_cur[1][:N]]
    # latent_true = latent_true[:N]

    with torch.no_grad():
        model._update_factors(observations_cur, full=True)
        model._update_variational(observations_cur, full=True)

elif cur_eval == 'train':

    observations_cur = observations_train
    latent_true = torch.tensor(true_latent_train).to("cpu")

    # observations_cur = [observations_cur[0][:N], observations_cur[1][:N]]
    # latent_true = latent_true[:N]

    with torch.no_grad():
        model._update_factors(observations_cur)
        model._update_variational(observations_cur)

    epoch_batch = model.epoch_batch
    batch_id = model.mini_batches[epoch_batch[0]][epoch_batch[1]]
    latent_true = torch.tensor(latent_true)[batch_id]

posterior_mean = model.variational.suff_stat_mean[0].detach().cpu().numpy()
label_hat = clf.predict(posterior_mean)
label_hat = svm_map[label_hat]

from sklearn import metrics

accuracy = (torch.tensor(label_hat) == torch.tensor(latent_true).to("cpu")).sum() / len(label_hat)
confusion_matrix = metrics.confusion_matrix(latent_true, label_hat)

plt.figure(figsize=(int(confusion_matrix.shape[0] / 2), int(confusion_matrix.shape[1] / 2)))
plot_confusion(confusion_matrix, label=label_basic)
plt.imshow(confusion_matrix)
plt.title('Accuracy ' + cur_eval + ' ' + str(np.round(accuracy, 4)))
plt.xticks(np.unique(latent_true), filtered_labels[np.unique(latent_true)], rotation=25)
plt.yticks(np.unique(latent_true), filtered_labels[np.unique(latent_true)], rotation=0)
print()

# %%
# %%


coef_matrix = np.abs(clf.coef_)
coef_matrix = coef_matrix / np.expand_dims(np.max(coef_matrix, axis=1), axis=1)

plt.figure()
plt.imshow(coef_matrix, cmap='inferno', aspect='auto')
plt.colorbar()
plt.xlabel('Latent Dimension')
plt.ylabel('Coefficient Amplitude')

plt.figure(figsize=(8 * 4, 4))
for ii in range(8):
    plt.subplot(1, 8, 1 + ii)
    plt.plot(coef_matrix[ii, :])

# %%

clf.classes_

# %%


colors = colors_basic[latent_true]
# colors = colors_categories[latent_true]
# colors = translation_color
# colors = rotation_color

plot_width = dim_latent - 1
plot_heigh = dim_latent - 1
plt.figure(figsize=(5 * plot_width, 5 * plot_heigh))
for ii in np.arange(dim_latent):
    for jj in np.arange(ii + 1, dim_latent):
        plt.subplot(plot_heigh, plot_width, jj + ii * (dim_latent - 1))
        plt.scatter(posterior_mean[:, jj], posterior_mean[:, ii], c=colors)
        plt.xlabel('Dim ' + str(jj))
        plt.ylabel('Dim ' + str(ii))

        if ii == 0 and jj == 1:
            plt.legend()

for class_id in range(len(colors_basic)):
    plt.scatter(0, 0, color=colors_basic[class_id], label=label_basic[class_id])
plt.legend()

# %%


img_tensor = mean + sttd * torch.randn(H, W, device=device)
img_tensor = torch.tensor(img_tensor, requires_grad=True)
#img_tensor = torch.tensor(observations_cur[0][0].unsqueeze(dim=0), requires_grad=True)

plt.figure()
plt.imshow(img_tensor.detach().numpy())
# %%








plt.imshow(f_image.numpy().squeeze())
plt.show()


#%%

import torch.nn.functional as F
# Creating a 4x4 mean filter. It needs to have (batches, channels, filter height, filter width)
t_filter = torch.as_tensor(np.full((1, 1, 10, 10), 0.001), device=device, dtype=dtype)
# Using F.conv2d to apply the filter


import torchvision
neural_net = model.recognition_factors[0]

mean = observations_train[0].mean()
sttd = 10 * observations_train[0].std()

# mean = 0
# sttd = 1

# We now freeze the parameters of our pretrained model
for param in neural_net.parameters():
    param.requires_grad_(False)

# generating the initial image with random pixel values between 0 and 1 with
# requires_grad True and values added to optimizer
H = observations_train[0].shape[-2]  # height of input image
W = observations_train[0].shape[-1]  # width of input image
img_tensor = mean + sttd * torch.randn(20, 1, H, W, device=device)
img_tensor = torch.tensor(F.conv2d(img_tensor, t_filter, padding='same'), requires_grad=True)
#img_tensor = torch.tensor(img_tensor, requires_grad=True)


#img_tensor = torch.tensor(observations_cur[0][0].unsqueeze(dim=0))
#img_tensor = torch.tensor(img_tensor.repeat(10, 1, 1, 1), requires_grad=True)

# optimizer = torch.optimizer.Adam()
lr = 0.1
optimizer = torch.optim.SGD([img_tensor], lr=lr)

catid = 4
num_epochs = 2000
display_every = 100

loss_tot = []
for epoch in range(num_epochs):

    optimizer.zero_grad()

    img_tensor2 = F.conv2d(img_tensor, t_filter, padding='same')
    latent_mean = neural_net(img_tensor2)[:, :dim_latent]

    mask = latent_true == catid
    centroid = torch.tensor(posterior_mean[mask].mean(axis=0), device=device)

    loss = torch.sum((centroid - latent_mean) ** 2)
    loss.backward()
    optimizer.step()
    loss_tot.append(loss.item())


    #with torch.no_grad():
    #img_tensor = torch.tensor(F.conv2d(img_tensor, t_filter, padding='same'), requires_grad=True)
    #optimizer = torch.optim.SGD([img_tensor], lr=lr)



    if epoch % display_every == 0:
        print('epoch: {}/{}, activation: {}'.format(epoch, num_epochs, loss))
        toplot = img_tensor.mean(dim=0).squeeze(dim=0).cpu().detach().numpy()
        plt.figure()
        plt.imshow(toplot)
        plt.title('input image after {} epochs'.format(epoch))

#%%

plt.figure()
plt.plot(loss_tot)

#%%

print('epoch: {}/{}, activation: {}'.format(epoch, num_epochs, loss))
toplot = img_tensor.mean(dim=0).squeeze(dim=0).cpu().detach().numpy()
plt.figure()
plt.imshow(toplot)
plt.title('input image after {} epochs'.format(epoch))
plt.show()

# %%

with torch.no_grad():
    img_tensor = gaussian_blur(img_tensor)
img_tensor = torch.tensor(img_tensor, requires_grad=True)
toplot = img_tensor.mean(dim=0).squeeze(dim=0).cpu().detach().numpy()
plt.imshow(toplot)
plt.title('input image after {} epochs'.format(epoch))
plt.show()

do_load = True
from save_load import load_crpm

if do_load:
    name = './low_variation_crpm.pkl'
    model_loaded, observations, true_latent = load_crpm(name)
    model = model_loaded
    model.recognition_variational.to(device)
    [i.to(device) for i in model.recognition_factors]
    dim_latent = model.dim_latent

# %% md

# Plot Loss

# %%

model.recognition_variational.to(device)
[i.to(device) for i in model.recognition_factors]
print('Moved To GPU')

[ii.eval() for ii in model.recognition_factors]
model.recognition_variational.eval()
print('Eval Mode')

plt.figure()
plt.plot(model.loss_tot, c='k')
plt.xlabel('epoch')
plt.ylabel('- Free Energy')

# %% md

# Display Latent MAP

# %%

cur_eval = 'train'

if cur_eval == 'test':
    observations_cur = observations_tests
    latent_true = true_latent_tests

elif cur_eval == 'train':
    observations_cur = observations_train
    latent_true = true_latent_train

with torch.no_grad():
    # observations_cur = [i.to("cpu") for i in observations_cur]
    # latent_true = torch.tensor(latent_true)
    model._update_factors(observations_cur)
    model._update_variational(observations_cur)

    epoch_batch = model.epoch_batch
    # epoch_batch = [0, 0]
    batch_id = model.mini_batches[epoch_batch[0]][epoch_batch[1]]
    latent_true = torch.tensor(latent_true)[batch_id]

# J = 1
# posterior_mean = model.factors.suff_stat_mean[0][J].detach().cpu().numpy()
posterior_mean = model.variational.suff_stat_mean[0].detach().cpu().numpy()

translation_color = translation_train - translation_train.mean()
translation_color = translation_color / translation_color.abs().max()
translation_color = 0.5 * (translation_color + 1)
translation_color = torch.cat((translation_color, 0.5 * torch.ones(translation_color.shape[0], 1)), dim=-1)

rotation_color = rotation_train - rotation_train.mean()
rotation_color = rotation_color / rotation_color.abs().max()
rotation_color = 0.5 * (rotation_color + 1)

colors = colors_basic[latent_true]
# colors = colors_categories[latent_true]
# colors = colors_categories[true_identity_train]


# colors = translation_color[batch_id]
# colors = rotation_color

plot_width = dim_latent
plot_heigh = dim_latent
plt.figure(figsize=(5 * plot_width, 5 * plot_heigh))
for ii in np.arange(dim_latent):
    # for ii in np.arange(1):
    for jj in np.arange(ii + 1, dim_latent):
        plt.subplot(plot_heigh, plot_width, jj + ii * plot_width)
        plt.scatter(posterior_mean[:, jj], posterior_mean[:, ii], c=colors)
        plt.xlabel('Dim ' + str(jj))
        plt.ylabel('Dim ' + str(ii))

for class_id in range(len(colors_basic)):
    plt.scatter(0, 0, color=colors_basic[class_id], label=label_basic[class_id])
plt.legend()
plt.show()

# %%

from sklearn.cluster import KMeans

X = posterior_mean.astype('float64')

num_centroid = 15
num_neighboo = 10
kmeans = KMeans(n_clusters=num_centroid, random_state=0).fit(X)

centroids = 1.2 * kmeans.cluster_centers_
centroids_radius = np.argsort(-(centroids ** 2).sum(axis=-1))
centroids = centroids[centroids_radius]

# %%

# plot_width = dim_latent - 1
# plot_heigh = dim_latent - 1
# plt.figure(figsize =(5 * plot_width, 5 * plot_heigh))
# for ii in np.arange(dim_latent):
#     for jj in np.arange(ii + 1, dim_latent):
#         plt.subplot(plot_heigh, plot_width, jj + ii * (dim_latent-1))
#         plt.scatter(posterior_mean[:, jj], posterior_mean[:, ii], c=colors)
#         plt.scatter(centroids[:, jj], centroids[:, ii], c='k', s=300)
#         plt.xlabel('Dim ' + str(jj))
#         plt.ylabel('Dim ' + str(ii))

# %%

plt.figure(figsize=(3 * (num_neighboo + 1), 3 * num_centroid))

for jj in range(num_centroid):
    centroid = centroids[jj]
    dists = ((torch.tensor(posterior_mean) - centroid) ** 2).sum(dim=-1)
    _, indices = dists.sort()
    offset = 10
    indices = indices[offset:(offset + num_neighboo)]

    for ii in range(num_neighboo):

        id = indices[ii]
        id = batch_id[id]
        plt.subplot(num_centroid, len(indices) + 1, ii + 1 + jj * (num_neighboo + 1))
        # lab_basic = label_basic[latent_true[id]]

        plt.imshow(observations_cur[0][id, 0].to("cpu"), cmap='gray', aspect='auto')
        # plt.title(lab_basic)
        plt.xticks([])
        plt.yticks([])

        if ii == 0:
            plt.ylabel('Centroid ' + str(jj))

    plt.subplot(num_centroid, len(indices) + 1, ii + 2 + jj * (num_neighboo + 1))
    plt.scatter(posterior_mean[:, -1], posterior_mean[:, -2], c=colors)
    plt.scatter(centroids[jj, -1], centroids[jj, -2], c='k', s=300)

# %% md

# Train an SVM on the trained latent space. Test it on unseen data

# %%

from sklearn import svm

cur_eval = 'train'
if cur_eval == 'test':

    observations_cur = observations_tests
    latent_true = torch.tensor(true_latent_tests).to("cpu")

    # observations_cur = [observations_cur[0][:N], observations_cur[1][:N]]
    # latent_true = latent_true[:N]

    with torch.no_grad():
        model._update_factors(observations_cur, full=True)
        model._update_variational(observations_cur, full=True)


elif cur_eval == 'train':

    observations_cur = observations_train
    latent_true = torch.tensor(true_latent_train).to("cpu")

    # observations_cur = [observations_cur[0][:N], observations_cur[1][:N]]
    # latent_true = latent_true[:N]

    with torch.no_grad():
        model._update_factors(observations_cur)
        model._update_variational(observations_cur)

    epoch_batch = model.epoch_batch
    batch_id = model.mini_batches[epoch_batch[0]][epoch_batch[1]]
    latent_true = torch.tensor(latent_true)[batch_id]

posterior_mean = model.variational.suff_stat_mean[0].detach().cpu().numpy()

# Train SVM
svm_map, latent_true_svm = np.unique(latent_true, return_inverse=True)
# clf = svm.SVC(C=10,kernel='linear')
clf = svm.LinearSVC(C=50, max_iter=100000, tol=0.000001)

clf.fit(posterior_mean, latent_true_svm)

cur_eval = 'test'
if cur_eval == 'test':

    observations_cur = observations_tests
    latent_true = torch.tensor(true_latent_tests).to("cpu")

    # observations_cur = [observations_cur[0][:N], observations_cur[1][:N]]
    # latent_true = latent_true[:N]

    with torch.no_grad():
        model._update_factors(observations_cur, full=True)
        model._update_variational(observations_cur, full=True)

elif cur_eval == 'train':

    observations_cur = observations_train
    latent_true = torch.tensor(true_latent_train).to("cpu")

    # observations_cur = [observations_cur[0][:N], observations_cur[1][:N]]
    # latent_true = latent_true[:N]

    with torch.no_grad():
        model._update_factors(observations_cur)
        model._update_variational(observations_cur)

    epoch_batch = model.epoch_batch
    batch_id = model.mini_batches[epoch_batch[0]][epoch_batch[1]]
    latent_true = torch.tensor(latent_true)[batch_id]

posterior_mean = model.variational.suff_stat_mean[0].detach().cpu().numpy()
label_hat = clf.predict(posterior_mean)
label_hat = svm_map[label_hat]

from sklearn import metrics

accuracy = (torch.tensor(label_hat) == torch.tensor(latent_true).to("cpu")).sum() / len(label_hat)
confusion_matrix = metrics.confusion_matrix(latent_true, label_hat)

plt.figure(figsize=(int(confusion_matrix.shape[0] / 2), int(confusion_matrix.shape[1] / 2)))
plot_confusion(confusion_matrix, label=label_basic)
plt.imshow(confusion_matrix)
plt.title('Accuracy ' + cur_eval + ' ' + str(np.round(accuracy, 4)))
plt.xticks(np.unique(latent_true), filtered_labels[np.unique(latent_true)], rotation=25)
plt.yticks(np.unique(latent_true), filtered_labels[np.unique(latent_true)], rotation=0)
print()

# %%


# %%

np.unique(latent_true_svm)

# %%

clf.decision_function(posterior_mean).shape

# %%

coef_matrix.max()

# %%


coef_matrix = np.abs(clf.coef_)
coef_matrix = coef_matrix / np.expand_dims(np.max(coef_matrix, axis=1), axis=1)

plt.figure()
plt.imshow(coef_matrix, cmap='inferno', aspect='auto')
plt.colorbar()
plt.xlabel('Latent Dimension')
plt.ylabel('Coefficient Amplitude')

plt.figure(figsize=(8 * 4, 4))
for ii in range(8):
    plt.subplot(1, 8, 1 + ii)
    plt.plot(coef_matrix[ii, :])

# %%

clf.classes_

# %%


colors = colors_basic[latent_true]
# colors = colors_categories[latent_true]
# colors = translation_color
# colors = rotation_color

plot_width = dim_latent - 1
plot_heigh = dim_latent - 1
plt.figure(figsize=(5 * plot_width, 5 * plot_heigh))
for ii in np.arange(dim_latent):
    for jj in np.arange(ii + 1, dim_latent):
        plt.subplot(plot_heigh, plot_width, jj + ii * (dim_latent - 1))
        plt.scatter(posterior_mean[:, jj], posterior_mean[:, ii], c=colors)
        plt.xlabel('Dim ' + str(jj))
        plt.ylabel('Dim ' + str(ii))

        if ii == 0 and jj == 1:
            plt.legend()

for class_id in range(len(colors_basic)):
    plt.scatter(0, 0, color=colors_basic[class_id], label=label_basic[class_id])
plt.legend()

# %%

import torchvision

img_tensor = mean + sttd * torch.randn(1, 1, H, W, device=device)
gaussian_blur = torchvision.transforms.GaussianBlur(5, sigma=0.1)

plt.figure()
toplot = img_tensor.mean(dim=0).squeeze(dim=0).cpu().detach().numpy()
plt.imshow(toplot)

plt.figure()
toplot = gaussian_blur(img_tensor).mean(dim=0).squeeze(dim=0).cpu().detach().numpy()
plt.imshow(toplot)

# %%


neural_net = model.recognition_factors[0]

mean = observations_train[0].mean()
sttd = 10 * observations_train[0].std()

# mean = 0
# sttd = 1

# We now freeze the parameters of our pretrained model
for param in neural_net.parameters():
    param.requires_grad_(False)

# generating the initial image with random pixel values between 0 and 1 with
# requires_grad True and values added to optimizer
H = observations_train[0].shape[-2]  # height of input image
W = observations_train[0].shape[-1]  # width of input image
img_tensor = mean + sttd * torch.randn(600, 1, H, W, device=device)
img_tensor = torch.tensor(img_tensor, requires_grad=True)

# optimizer = torch.optimizer.Adam()
optimizer = torch.optim.Adam([img_tensor], lr=0.0005)

catid = 4
num_epochs = 500000
display_every = 1000
unit_idx = 11  # unit of the convolution layer that we wish to visualize
gaussian_blur = torchvision.transforms.GaussianBlur(1)

loss_tot = []
for epoch in range(num_epochs):

    optimizer.zero_grad()
    latent_mean = neural_net(img_tensor)[:, :dim_latent]

    mask = latent_true == catid
    centroid = torch.tensor(posterior_mean[mask].mean(axis=0), device=device)

    loss = torch.sum((centroid - latent_mean) ** 2)
    loss.backward()
    optimizer.step()
    loss_tot.append(loss.item())

    # normalize the updated img_tensor to have pytorch specified mean and std. dev.
    # img_tensor = normalize(img_tensor.clone().detach()).requires_grad_(True)
    # the above step of renormalizing the updated img_tensor does not work: the activation remains frozen at ~ 2.368 and no patterns seem to develop in the image
    # Will have to understand this further

    if epoch % display_every == 0:
        print('epoch: {}/{}, activation: {}'.format(epoch, num_epochs, loss))
        toplot = img_tensor.mean(dim=0).squeeze(dim=0).cpu().detach().numpy()
        plt.imshow(toplot)
        plt.title('input image after {} epochs'.format(epoch))
        plt.show()

#         with torch.no_grad():
#            img_tensor = gaussian_blur(img_tensor)
#         img_tensor = torch.tensor(img_tensor, requires_grad=True)

plt.figure()
plt.plot(loss_tot)

# %%

img_tensor.shape

# %%

img_tensor.shape

# %%

with torch.no_grad():
    img_tensor = gaussian_blur(img_tensor)
img_tensor = torch.tensor(img_tensor, requires_grad=True)
toplot = img_tensor.mean(dim=0).squeeze(dim=0).cpu().detach().numpy()
plt.imshow(toplot)
plt.title('input image after {} epochs'.format(epoch))
plt.show()
