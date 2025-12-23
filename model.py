from sklearn.svm import SVC
from pathutils.explorationmetrics import calculate_time_metrics
from pathutils.spectral_analysis import calculate_frequency_metrics
from collections.abc import Callable


class BirdClassifier:
    def __init__(self):
        self.model = SVC()
    
    def train(self, x, y):
        self.model.fit(x, y)

    def classify(self, data):
        self.model.predict(data)


def feature_extraction(data, sampling_rates, feature_list: list[str]):
    """
    features[class][sample][feature], targets[class][sample]   
    """
    time_features, time_targets = calculate_time_metrics(data, feature_list)
    frequency_features, frequency_targets = calculate_frequency_metrics(data, sampling_rates, feature_list)
    
    features = time_features + frequency_features
    targets = time_targets + frequency_targets
    
    return features, targets
        