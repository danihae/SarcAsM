import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import label, distance_transform_edt
from skimage.morphology import binary_dilation, remove_small_holes, binary_closing


def simulate_training_data(folder, input_len=512, n=100, freq_range=(0.04, 0.25), prob_zeros=0.5,
                           clip_thrs=(-0.75, 0.75), random_drift_amp_range=(0.005, 0.02),
                           random_drift_freq_range=(0, 0.05), noise_amp_range=(0, 0.25), plot=False):
    os.makedirs(folder, exist_ok=True)

    for i in range(n):

        freq = np.random.uniform(freq_range[0], freq_range[1])
        x_range = np.arange(input_len)
        amp_mod = 1 + np.abs(np.cos(x_range * np.random.uniform(0.01, 0.02)))
        if np.random.binomial(1, prob_zeros):
            freq = 0

        y_sim = np.clip(np.cos(x_range * freq), None, np.random.uniform(clip_thrs[0], clip_thrs[1]))
        y_sim -= np.max(y_sim)
        y_sim = amp_mod * y_sim

        # calculate systole
        y_systole = np.zeros_like(y_sim)
        y_systole[y_sim != 0] = 1

        # add random drift
        random_drift = (np.random.uniform(random_drift_amp_range[0], random_drift_amp_range[1]),
                        np.random.uniform(random_drift_freq_range[0], random_drift_freq_range[1]))
        y_sim += random_drift[0] * np.cos(random_drift[1] * x_range)

        # add normal noise
        y_sim += np.random.normal(0, np.random.uniform(noise_amp_range[0], noise_amp_range[1]), size=input_len)

        if plot:
            plt.figure()
            plt.plot(x_range, y_sim)
            plt.plot(x_range, y_systole)
            plt.show()

        np.savetxt(folder + f'simulated_{i}.txt', y_sim)
        np.savetxt(folder + f'simulated_{i}_systole.txt', y_systole)


def plot_selection_training_data(dataset, n_sample):
    selection = dataset.data[np.random.choice(dataset.data.shape[0], n_sample)]
    fig, axs = plt.subplots(figsize=(5, 2 * n_sample), nrows=n_sample)

    for i, d_i in enumerate(selection):
        axs[i].plot(d_i[0], c='k')
        axs[i].plot(d_i[1], c='r')

    plt.show()


def find_txt_files(root_dir):
    txt_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(dirpath, filename))
    return txt_files


def get_device(print_device=False):
    """
    Determines the most suitable device (CUDA, MPS, or CPU) for PyTorch operations.

    Returns:
    - A torch.device object representing the selected device.
    """
    if torch.backends.cuda.is_built():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_built():  # only for Apple M1/M2/...
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("Warning: No CUDA or MPS device found. Calculations will run on the CPU, "
              "which might be slower.")
    if print_device:
        print(f"Using device: {device}")
    return device


def distance_transform(input, target):
    """
    Compute a normalized distance transform for each labeled region in the target.

    The function calculates the Euclidean distance transform for each unique label
    in the target array. Each distance transform is then normalized by its maximum
    value to ensure distances are scaled between 0 and 1. These normalized distance
    transforms are summed to produce a composite distance map.

    Parameters
    ----------
    input : ndarray
        The input image array. This parameter is currently not used in the function,
        but included for future extensions or modifications.
    target : ndarray
        The target image array containing labeled regions. The regions should be
        labeled as distinct integers, with background typically labeled as 0.

    Returns
    -------
    distances : ndarray
        An array of the same shape as `target`, containing the normalized distance
        transform values for each labeled region in `target`.

    Notes
    -----
    This implementation assumes that the target contains non-overlapping labeled
    regions. Overlapping regions will result in undefined behavior.
    """
    # Initialize an array to store the distance transforms
    distances = np.zeros_like(input, dtype=float)
    # Label the connected components in the target
    labels, n_labels = label(target)
    # Iterate over each label to compute its distance transform
    for label_i in range(1, n_labels + 1):
        # Create a binary mask for the current label
        labels_i = labels == label_i
        # Compute the Euclidean distance transform for the current label
        distance_i = distance_transform_edt(labels_i)
        # Normalize and accumulate the distance transform
        distances += distance_i / distance_i.max()
    return distances


def process_contractions(contractions, signal=None, threshold=0.05, area_min=3, dilate_surrounding=2, len_min=4,
                         merge_max=3):
    """
    Process contraction time series to filter based on specified criteria.

    Parameters
    ----------
    contractions : ndarray
        Array indicating intervals of potential contractions (output of ContractionNet).
    signal : ndarray
        Array of the original signal values corresponding to contractions. If None, no signal will be processed.
    threshold : float, optional
        Threshold value to binarize contractions, by default 0.05.
    area_min : int, optional
        Minimum area under the signal curve for contraction interval to consider, by default 3 frames.
    dilate_surrounding : int, optional
        Number of frames to dilate around each contraction for offset calculation, by default 2 frames.
    len_min : int, optional
        Minimum length of a contraction to consider, by default 4 frames.
    merge_max : int, optional
        Maximum gap to merge subsequent filtered contractions, by default 3 frames.

    Returns
    -------
    ndarray
        Binary array with filtered contractions based on the specified criteria.
    """
    contr_labels, n_labels = label(contractions > threshold)

    contr_labels_filtered = np.zeros_like(contr_labels)

    for i, contr_i in enumerate(np.unique(contr_labels)[1:]):
        contr_labels_i = contr_labels == contr_i
        len_i = np.count_nonzero(contr_labels_i)
        if signal is not None:
            signal_i = signal[contr_labels_i]
            # Dilate contraction labels and calculate offset
            contr_labels_i_dilated = binary_dilation(contr_labels_i, np.ones((dilate_surrounding * 2 + 1)))
            surrounding_i = contr_labels_i_dilated ^ contr_labels_i
            offset_i = np.mean(signal[surrounding_i])
            signal_i -= offset_i
            # Calculate area under peak and length
            area_i = np.abs(np.sum(signal_i))
        else:
            area_i = np.inf

        if area_i >= area_min and len_i >= len_min:
            contr_labels_filtered += contr_labels_i

    # Remove small holes
    contr_labels_filtered = binary_closing(contr_labels_filtered, np.ones(merge_max))

    return contr_labels_filtered
