import os
import librosa as lr
import numpy as np
import pandas as pd
from os import PathLike


def load_audio_data(base_path: str | bytes| PathLike ='./Data') -> pd.DataFrame:
    """
    Loads audio data from the specified base path and returns it as a numpy array.

    The base path should contain folders, each containing audio files.
    The function will iterate over each folder and load the audio files into a numpy array.

    Parameters
    ----------
    None

    Returns
    -------
    raw_data : DataFrame
        A pandas DataFrame containing the loaded audio data.
    sampling_rates : DataFrame
        A pandas DataFrame containing the sampling rates (in Hz) of the audio data.
    """

    classes = os.listdir(base_path)
    if '.DS_Store' in classes:
        classes.remove('.DS_Store')

    raw_data = [[] for _ in range(len(classes))]    # initialize list to hold raw audio data
    sampling_rates = [[] for _ in range(len(classes))]  # initialize list to hold sampling rates
    class_count = 0
    cls = []

    for path in classes:
        full_path = os.path.join(base_path, path)           # full path to class folder
        cls.append(path)
        if os.path.isdir(full_path):
            dir = os.listdir(full_path)         # list all files in directory

            if '.DS_Store' in dir:          # remove .DS_Store files for mac users
                os.remove(os.path.join(full_path, '.DS_Store'))
                dir.remove('.DS_Store')

            for sample in range(3):            # iterate through samples in class folder
                file_data, sr = lr.load(os.path.join(full_path, dir[sample]), sr=None)        # load audio file
                
                raw_data[class_count].append(file_data)         # append data to list
                sampling_rates[class_count].append(sr)          # append sampling rate to list
        class_count += 1

    raw_data = pd.DataFrame(raw_data, index=cls)       # rows are classes, columns are samples 
    sampling_rates = pd.DataFrame(sampling_rates)
    return raw_data, sampling_rates


if __name__ == "__main__":
    print("This is a utilities file and should not be run as the main file.")
