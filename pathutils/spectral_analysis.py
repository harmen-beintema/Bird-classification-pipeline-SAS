"""
spectral analysis using the following techniques:

- Power spectral density.
- Spectrogram.
- Average power.
- Max/Min frequency.
- Min/Max Magnitude (or power).
"""
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt


def spectral_density(signal: ArrayLike, sampling_rate: int) -> np.ndarray:
    """
    Compute the Power Spectral Density (PSD) of a signal.
    
    Args:
        input:
            signal : ArrayLike
                An array containing the amplitudes of a signal for each timestep.
            sampling_rate : int
                The sampling rate of the signal
        output : np.ndarray
                An array containing the power spectral density of the signal
    """
    t = np.arange(len(signal)) / sampling_rate
    n = len(t)
    fft = np.fft.fft(signal, n=len(t)) 
    return fft * np.conj(fft) / n
    

def spectrogram(signal: ArrayLike, sampling_rate: int, overlap=400, NFFT: int=5000) -> None:
    """
    Plot a spectrogram of a signal
    Args:
        input:
            signal : ArrayLike
                An array containing the amplitudes of a signal for each timestep.
            sampling_rate : int
                The sampling rate of the signal
    """
    plt.specgram(signal[0:1000], NFFT, Fs=sampling_rate, noverlap=overlap, cmap='jet_r')
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def plot_psd(signal: ArrayLike, sampling_rate: int, first_half: bool) -> None:
    """
    Plot the psd of a signal
    Args:
        input:
            signal : ArrayLike
                An array containing the amplitudes of a signal for each timestep.
            sampling_rate : int
                The sampling rate of the signal
    """
    psd = spectral_density(signal, sampling_rate)
    period = 1 / sampling_rate
    n = len(signal)
    frequencies = (1 / (period * n)) * np.arange(n)
    
    if first_half:
        L = np.arange(1, np.floor(n/2), dtype=int)
    else:
        L = np.arange(1, n, dtype=int)

    plt.plot(frequencies[L], psd[L], color='r', linewidth=1, label='Noisy')
    plt.xlim(frequencies[L[0]], frequencies[L[-1]])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()
    plt.show()


def average_power(signal: ArrayLike, sampling_rate: int) -> float:
    """
    Return the average power of the signal.    
    
    Args:
        input:
            signal : ArrayLike
                An array containing the amplitudes of a signal for each timestep.
            sampling_rate : int
                The sampling rate of the signal
        output:
            mean: float
                The mean average power of the signal
    """
    return np.mean(spectral_density(signal, sampling_rate))


def min_frequency(signal: ArrayLike, sampling_rate: int) -> int:
    """
    Return the min frequency of the signal.    
    
    Args:
        input:
            signal : ArrayLike
                An array containing the amplitudes of a signal for each timestep.
            sampling_rate : int
                The sampling rate of the signal
        output:
            min : int
                The minimum frequency in the signal
    """
    t = np.arange(len(signal)) / sampling_rate
    n = len(t)
    fft = np.fft.fft(signal, n=len(t))      # fast fourier transform reveals the frequencies
    return np.min(fft)


def max_frequency(signal: ArrayLike, sampling_rate: int) -> int:
    """
    Return the max frequency of the signal.    
    
    Args:
        input:
            signal : ArrayLike
                An array containing the amplitudes of a signal for each timestep.
            sampling_rate : int
                The sampling rate of the signal
        output:
            max : int
                The maximum frequency in the signal
    """
    t = np.arange(len(signal)) / sampling_rate
    n = len(t)
    fft = np.fft.fft(signal, n=len(t))      # fast fourier transform reveals frequencies
    return np.max(fft)


def min_power(signal: ArrayLike, sampling_rate: int) -> float:
    """
    Return the min power of the signal.
    
    Args:
        input:
            signal : ArrayLike
                An array containing the amplitudes of a signal for each timestep.
            sampling_rate : int
                The sampling rate of the signal
        output:
            min : float
                The min power of the signal
    """
    psd = spectral_density(signal, sampling_rate)
    return np.min(psd)


def max_power(signal: ArrayLike, sampling_rate: int) -> float:
    """
    Return the max power of the signal.
    
    Args:
        input:
            signal : ArrayLike
                An array containing the amplitudes of a signal for each timestep.
            sampling_rate : int
                The sampling rate of the signal
        output:
            max : float
                The max power of the signal
    """
    psd = spectral_density(signal, sampling_rate)
    return np.max(psd)


def calculate_frequency_metrics(data: pd.DataFrame, sampling_rates: pd.DataFrame, feature_list: list[str]) -> tuple[list, list]:
    """
    Compute all frequency metrics for a given signal.
    
    Args:
        input:
            data : pd.DataFrame
                A dataframe containing the amplitudes of a signal for each timestep.
            sampling_rates : pd.DataFrame
                A dataframe containing the sampling rates of the samples
        output:
            results : list[feature][metric]
                A list containing all computed metrics.
            targets : list[feature][target]
                A list containing the target class for each metric
    """
    metrics = {
        "Spectral Density": spectral_density,
        "Average Power": average_power,
        "Min Frequency": min_frequency,
        "Max Frequency": max_frequency,
        "Min Power": min_power,
        "Max Power": max_power
    }
    
    results = [[] for _ in range(len(feature_list))]
    targets = []

    for feature in range(len(feature_list)):        # for each chosen feature
        if feature in metrics.keys():       # if it is a frequency feature
            for cls in range(len(data)):        # for each class in the dataset
                for sample in range(len(data.iloc[cls])):       # for each sample in that class
                    results[feature].append(metrics[feature](data.iloc[cls, sample],
                                                              sampling_rates.iloc[cls, sample]))     # calculate the features
                    targets.append(cls)     # add target class at same index
    return results, targets


if __name__ == "__main__":
    print("This is a utilities file and should not be run as the main file.")