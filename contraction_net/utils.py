import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import label, distance_transform_edt
from skimage.morphology import binary_dilation, remove_small_holes, binary_closing


def simulate_training_data(folder, input_len=512, n=100, freq_range=(0.04, 0.25), prob_zeros=0.5,
                           clip_thrs=(-0.75, 0.75), random_drift_amp_range=(0.005, 0.02),
                           random_drift_freq_range=(0, 0.05), noise_amp_range=(0, 0.25), plot=False):
    """
    Simulate training data for ContractionNet with sinusoidal patterns, random drift, and noise.

    Parameters
    ----------
    folder : str
        Path to the folder where the simulated data will be saved.
    input_len : int, optional
        Length of each simulated sequence, by default 512.
    n : int, optional
        Number of sequences to simulate, by default 100.
    freq_range : tuple of float, optional
        Range of frequencies for the sinusoidal patterns, by default (0.04, 0.25).
    prob_zeros : float, optional
        Probability of zero frequency (flat line), by default 0.5.
    clip_thrs : tuple of float, optional
        Clipping thresholds for the sinusoidal values, by default (-0.75, 0.75).
    random_drift_amp_range : tuple of float, optional
        Amplitude range for the random drift, by default (0.005, 0.02).
    random_drift_freq_range : tuple of float, optional
        Frequency range for the random drift, by default (0, 0.05).
    noise_amp_range : tuple of float, optional
        Amplitude range for the added noise, by default (0, 0.25).
    plot : bool, optional
        Whether to plot the simulated data, by default False.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    for i in range(n):
        # Randomly select a frequency within the specified range
        freq = np.random.uniform(freq_range[0], freq_range[1])
        x_range = np.arange(input_len)

        # Amplitude modulation with a slow cosine function
        amp_mod = 1 + np.abs(np.cos(x_range * np.random.uniform(0.01, 0.02)))

        # Set frequency to zero with a certain probability
        if np.random.binomial(1, prob_zeros):
            freq = 0

        # Generate a cosine wave and clip its values
        y_sim = np.clip(np.cos(x_range * freq), None, np.random.uniform(clip_thrs[0], clip_thrs[1]))
        y_sim -= np.max(y_sim)  # Normalize to zero baseline
        y_sim = amp_mod * y_sim

        # Calculate systole as a binary mask where the signal is non-zero
        y_contr = np.zeros_like(y_sim)
        y_contr[y_sim != 0] = 1

        # Add random drift to the signal
        random_drift = (np.random.uniform(random_drift_amp_range[0], random_drift_amp_range[1]),
                        np.random.uniform(random_drift_freq_range[0], random_drift_freq_range[1]))
        y_sim += random_drift[0] * np.cos(random_drift[1] * x_range)

        # Add normal noise to the signal
        y_sim += np.random.normal(0, np.random.uniform(noise_amp_range[0], noise_amp_range[1]), size=input_len)

        # Plot the simulated data
        if plot:
            plt.figure()
            plt.plot(x_range, y_sim)
            plt.plot(x_range, y_contr)
            plt.show()

        # Save the simulated signal and its contraction mask to text files
        np.savetxt(os.path.join(folder, f'simulated_{i}.txt'), y_sim)
        np.savetxt(os.path.join(folder, f'simulated_{i}_contr.txt'), y_contr)


def plot_selection_training_data(dataset, n_sample):
    """
    Plot a random selection of training data sequences.

    Parameters
    ----------
    dataset : object
        Dataset object containing the training data.
    n_sample : int
        Number of samples to plot.
    """
    # Randomly select n_sample sequences from the dataset
    selection = dataset.data[np.random.choice(dataset.data.shape[0], n_sample)]

    # Create subplots
    fig, axs = plt.subplots(figsize=(5, 2 * n_sample), nrows=n_sample)

    for i, d_i in enumerate(selection):
        # Plot the data sequences
        axs[i].plot(d_i[0], c='k')  # Plot the first sequence in black
        axs[i].plot(d_i[1], c='r')  # Plot the second sequence in red

    plt.show()


def find_txt_files(root_dir):
    """
    Find all .txt files in a directory and its subdirectories.

    Parameters
    ----------
    root_dir : str
        Root directory to search for text files.

    Returns
    -------
    list of str
        List of paths to the found text files.
    """
    txt_files = []
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                # Append the full path of text files to the list
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
