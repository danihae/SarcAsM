import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class TimeSeriesAnnotator:
    """
    Class to annotate contraction intervals in multiple time-series files and save the results.

    This class allows users to interactively annotate the start and end of contractions
    in time-series data by clicking on the plot. The user can press the left mouse button
    at the start of a contraction interval and release at the end of the contraction interval.
    The results are saved to a specified output directory.

    Parameters
    ----------
    file_list : list of str
        List of file paths to the time-series data files to be annotated.
    output_dir : str
        Directory where the output files with annotated contraction intervals will be saved.
    figsize : tuple, optional
        Figure size in inches.

    Methods
    -------
    load_next_file()
        Load the next time-series file for annotation.
    press_callback(event)
        Record the x-coordinate (time) where the mouse button is pressed.
    release_callback(event)
        Record the x-coordinate (time) where the mouse button is released.
    save_and_load_next(event)
        Save the current annotations and load the next time-series file.
    reset_annotations(event)
        Reset the current annotations for the current time-series file.

    Notes
    -----
    In an interactive Jupyter notebook, add the following before running the TimeSeriesAnnotator:

    ```
    matplotlib.use('nbagg')
    %matplotlib notebook
    ```
    """

    def __init__(self, file_list, output_dir, figsize=(11, 3.5)):
        self.file_list = [f for f in file_list if not f.endswith('_contr.txt')]
        self.output_dir = output_dir
        self.current_file_index = 0
        self.start_contraction = []
        self.end_contraction = []
        self.fig, self.ax = plt.subplots(figsize=figsize, constrained_layout=True)
        self.load_next_file()

        self.fig.canvas.mpl_connect('button_press_event', self.press_callback)
        self.fig.canvas.mpl_connect('button_release_event', self.release_callback)

        save_ax = self.fig.add_axes([0.85, 0.01, 0.1, 0.075])
        self.save_button = Button(save_ax, 'Save & Next', color='lightgoldenrodyellow', hovercolor='0.975')
        self.save_button.on_clicked(self.save_and_load_next)

        reset_ax = self.fig.add_axes([0.75, 0.01, 0.1, 0.075])
        self.reset_button = Button(reset_ax, 'Reset', color='lightcoral', hovercolor='0.975')
        self.reset_button.on_clicked(self.reset_annotations)

        plt.show()

    def load_next_file(self):
        """Load the next time-series file for annotation."""
        if self.current_file_index < len(self.file_list):
            file_path = self.file_list[self.current_file_index]
            self.data = np.loadtxt(file_path)
            self.ax.clear()
            self.ax.plot(self.data, c='k')
            self.ax.set_title(f'Annotating {os.path.basename(file_path)}')
            self.start_contraction = []
            self.end_contraction = []
            plt.draw()
        else:
            plt.close(self.fig)
            print('All files annotated.')

    def press_callback(self, event):
        """Record the x-coordinate (time) where the mouse button is pressed."""
        self.ax.axvline(event.xdata, c='r', linestyle=':')
        self.start_contraction.append(event.xdata)

    def release_callback(self, event):
        """Record the x-coordinate (time) where the mouse button is released."""
        self.ax.axvline(event.xdata, c='r', linestyle=':')
        self.ax.axvspan(self.start_contraction[-1], event.xdata, alpha=0.5, color='red')
        self.end_contraction.append(event.xdata)

    def save_and_load_next(self, event):
        """Save the current annotations and load the next time-series file."""
        if self.current_file_index < len(self.file_list):
            file_path = self.file_list[self.current_file_index]
            output_file = os.path.join(self.output_dir, f'{os.path.basename(file_path).split(".")[0]}_contr.txt')
            start_end_contraction = np.asarray([self.start_contraction, self.end_contraction])
            np.savetxt(output_file, start_end_contraction.T, header='start end', comments='', fmt='%f')
            print(f'Saved contractions for {os.path.basename(file_path)} to {output_file}')

            self.current_file_index += 1
            self.load_next_file()

    def reset_annotations(self, event):
        """Reset the current annotations for the current time-series file."""
        self.start_contraction = []
        self.end_contraction = []
        self.ax.clear()
        self.ax.plot(self.data, c='k')
        self.ax.set_title(f'Annotating {os.path.basename(self.file_list[self.current_file_index])}')
        plt.draw()


def split_trace(filepath, output_dir, chunk_size=512, p=1):
    """
    Split a time-series trace into smaller chunks and save them to separate files.

    Parameters
    ----------
    filepath : str
        Path to the input time-series data file.
    output_dir : str
        Directory where the output chunk files will be saved.
    chunk_size : int, optional
        The size of each chunk, by default 512.
    p : float, optional
        Probability of saving each chunk, by default 1 (save all chunks).
    """
    # Read the time-series data
    data = np.loadtxt(filepath)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate the number of chunks
    num_chunks = len(data) // chunk_size + (len(data) % chunk_size > 0)

    # Split the data into chunks of size `chunk_size` and save each chunk to a separate file
    for i in range(num_chunks):
        if random.random() < p:
            chunk = data[i * chunk_size:(i + 1) * chunk_size]
            np.savetxt(os.path.join(output_dir, f'{os.path.basename(filepath).split(".")[0]}_{i}.txt'), chunk)
