from pathutils.load_audio_data import load_audio_data
from pathutils.spectral_analysis import calculate_frequency_metrics
from pathutils.explorationmetrics import calculate_time_metrics
from model import BirdClassifier
import matplotlib.pyplot as plt 
import numpy as np
import plotly.express as px
import time as t


""""
limitations/notes:
    main:
        no main routine has been implemented.
        
    computational:
        I feel like the data loading (and the computation of all the features for)
        all the samples is inefficient/takes too long. 
        O(n) for data load and O(k * n * f) for feature calculation 
        where k is the number of classes, n the total number of samples and f the amount of features to be calculated. 
    
    spectrogram:
        The spectrogram function works better over shorter time intervals. 
        Do I need to take this into account somewhere or can I trust that 
        the user will use the function with appropriate sample lengths?
   
    filter design:
        I wonder what frequencies to take for the FIR filter 
        since all birds (and even samples) have different frequencies in their calls. 
        Wouldn't a PSD filter be easier/better?
   
    model: 
        The functionality of the model is untested and a wrapper class around sklearn's SVC.
        
    Documentation/style:
        the current documentation is sparse, outdated and incomplete.
        the current code does not conform to OOP principles/best practices.
"""