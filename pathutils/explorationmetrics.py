"""
A couple of metrics to explore the dataset. The following metrics are implemented:
- Root Mean Square Error (RMSE)
- Standard Deviation (STD)
- Crest Factor (CF)
- Average Amplitude (AA)
- Min/Max Amplitude (MMA)
"""
import numpy as np
import pandas as pd


def max_amplitude(signal: np.ndarray) -> float:
    """
    Compute the Maximum Amplitude of a signal.
    Args:
        input:
            signal : np.ndarray
                An array containing the amplitudes of a signal for each timestep.
        output:
            float : The maximum amplitude of the signal.
    """
    return np.max(np.abs(signal))


def min_amplitude(signal: np.ndarray) -> float:
    """
    Compute the Minimum Amplitude of a signal.
    
    Args:
        input:
            signal : np.ndarray
                An array containing the amplitudes of a signal for each timestep.
        output:
            float : The minimum amplitude of the signal.
    """
    return np.min(np.abs(signal))


def average_amplitude(signal: np.ndarray) -> float:
    """
    Compute the Average Amplitude (AA) of a signal.

    Args:
        input:
            signal : np.ndarray
                An array containing the amplitudes of a signal for each timestep.
        output:
            float : The average amplitude of the signal.
    """
    return np.mean(np.abs(signal))  


def compute_rmse(signal: np.ndarray) -> float:
    """
    Compute the Root Mean Square Error (RMSE) of a signal.
    
    Args:
        input:
            signal : np.ndarray
                An array containing the amplitudes of a signal for each timestep.
        output:
            float : The RMSE of the signal.
    """
    return np.sqrt(np.mean(signal**2))


def compute_std(signal: np.ndarray) -> float:
    """
    Compute the Standard Deviation (STD) of a signal.
    
    Args:
        input:
            signal : np.ndarray
                An array containing the amplitudes of a signal for each timestep.
        output:
            float : The standard deviation of the signal.
    """
    return np.std(signal)


def compute_crest_factor(signal: np.ndarray) -> float:
    """
    Compute the Crest Factor (CF) of a signal.
    
    Args:
        input:
            signal : np.ndarray
                An array containing the amplitudes of a signal for each timestep.
        output:
            float : The Crest factor of the signal.
    """
    peak_amplitude = max_amplitude(signal)
    rms_value = compute_rmse(signal)
    if rms_value == 0:                                  # if mean is zero, rms is zero
        return 0
    return peak_amplitude / rms_value


def calculate_time_metrics(data: pd.DataFrame, feature_list: list[str]) -> list:
    """
    Compute all metrics based on time for a given signal.
    
    Args:
        input:
            data : pd.DataFrame
                An dataframe containing the amplitudes of a signal for each timestep.
        output:
            results : list[feature][metric]
                A list containing all computed metrics.
            targets : list[feature][target]
                A list containing the target class for each metric
    """
    metrics = {
        "Max Amplitude": max_amplitude,
        "Min Amplitude": min_amplitude,
        "Average Amplitude": average_amplitude,
        "RMSE": compute_rmse,
        "STD": compute_std,
        "Crest Factor": compute_crest_factor
    }
    
    results = [[] for _ in range(len(feature_list))]
    targets = []

    for feature in range(len(feature_list)):
        if feature in metrics.keys():
            for cls in data.index.values:
                for sample in data.loc[cls]:
                    results[feature].append(metrics[feature](sample))
                    targets.append(cls)
    return results, targets


if __name__ == "__main__":
    print("This is a utilities file and should not be run as the main file.")