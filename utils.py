import torch
import numpy as np
from matplotlib import pyplot as plt

def get_minibatches(num_epoch, len_full, len_minibatch):
    """ Returns mini-batch indices """

    num_minibatch = int(np.ceil(len_full / len_minibatch))

    mini_batches = []
    for epoch in range(num_epoch):
        if len_full == len_minibatch:
            permutation_cur = list(np.arange(len_full))
        else:
            permutation_cur = list(np.random.permutation(np.arange(len_full)))
        mini_batch_cur = [np.sort(permutation_cur[i * len_minibatch:(i + 1) * len_minibatch])
                          for i in range(num_minibatch)]

        mini_batches.append(mini_batch_cur)

    return mini_batches


def minibatch_tupple(input, dim,  idx, device=None):
    """ Extract relevant minibatch from tupled multifactorial observations """
    idx = torch.tensor(idx, device=device)
    return tuple([torch.index_select(obsj, dim, idx) for obsj in input])


def plot_confusion(matrix, label = None, overlay_text = True, normalize= True,  **kwargs):

    # Normalize confusion matrix
    matrix = matrix / matrix.sum(axis=1, keepdims=True) if normalize else matrix

    # Plot Matrix
    plt.imshow(matrix, **kwargs)

    if any(label == None) == None:
        label = np.arange(matrix.shape[0])
    plt.xticks(np.arange(8), label, rotation=25)
    plt.yticks(np.arange(8), label, rotation=0)

    if overlay_text:
        # Add the text
        x_start = 0 - 0.5
        y_start = 0 - 0.5
        x_end = matrix.shape[0] - 0.5
        y_end = matrix.shape[1] - 0.5
        size = len(matrix)

        # Add the text
        jump_x = (x_end - x_start) / (2.0 * size)
        jump_y = (y_end - y_start) / (2.0 * size)
        x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
        y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = np.round(matrix[y_index, x_index], 2)
                text_x = x + jump_x
                text_y = y + jump_y
                plt.text(text_x, text_y, label, color='black', ha='center', va='center')


def optimizer_wrapper(param, optimizer_param):

    optimizer_name = optimizer_param['name']
    optimizer_param = optimizer_param['param']

    if optimizer_name == 'Adam':
        return torch.optim.Adam(param, **optimizer_param)

    elif optimizer_name == 'SGD':
        return torch.optim.SGD(param, **optimizer_param)

    elif optimizer_name == 'Adamax':
        return torch.optim.Adamax(param, **optimizer_param)

    elif optimizer_name == 'LBFGS':
        return torch.optim.LBFGS(param, **optimizer_param)

    elif optimizer_name == 'RMSprop':
        return torch.optim.RMSprop(param, **optimizer_param)

    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(param, **optimizer_param)

    else:
        raise NotImplementedError()


def print_loss(loss, epoch_id, epoch_num, pct=0.001):
    """ Simple logger"""
    str_epoch = 'Epoch ' + str(epoch_id) + '/' + str(epoch_num)
    str_loss = ' Loss: %.6e' % loss

    if epoch_num < int(1/pct) or epoch_id % int(epoch_num * pct) == 0:
        print(str_epoch + str_loss)
        

def get_modulator0(x, amp):
    return (x - 0.5) / (1 - 0.5) * amp + (x - 1) / (0.5 - 1) * 1


def get_modulator1(x, amp):
    y = x
    y[x > 0.5] = get_modulator0(x[x > 0.5], amp)
    y[x <= 0.5] = 1/get_modulator0(1 - x[x <= 0.5], amp)

    return y


def get_color(index_base, index_sub=None, cmap=None, amp=3):

    # Init Base categories Color Map
    if cmap is None:
        cmap = plt.cm.tab10(np.linspace(0, 1, 10))

    # Get Base categories Color
    num_categories_basic = len(np.unique(index_base))
    colors_basic = cmap[:num_categories_basic]

    # Get Subcategories Color
    if index_sub is None:
        colors_categories = None

    else:
        num_categories = len(np.unique(index_sub))
        colors_categories = np.zeros((num_categories, 4))

        for ii in range(num_categories_basic):

            # Select Subcategories
            mask = np.where(index_base == ii)

            # Get Indices and numbers
            sub_categories_index = index_sub[mask]
            sub_categories_index_unique = np.unique(sub_categories_index)
            sub_categories_index_num = len(sub_categories_index_unique)

            # Current base color
            color_basic_cur = colors_basic[ii]

            # Color Modulation Index
            modulation_index = np.expand_dims(get_modulator1(np.linspace(0, 1, sub_categories_index_num), amp=amp), axis=1)

            # Modulate base color
            modulated_color = color_basic_cur ** modulation_index
            modulated_color[:, 3] = 1
            colors_categories[sub_categories_index_unique] = modulated_color

    return colors_basic, colors_categories


def diagonalize(z):
    """ Use a batch vector to create diagonal batch matrices """
    Z = torch.zeros((*z.shape, z.shape[-1]), device=z.device, dtype=z.dtype)
    Z[..., range(z.shape[-1]), range(z.shape[-1])] = z
    return Z