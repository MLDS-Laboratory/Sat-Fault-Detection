import numpy as np
from anomaly_detection_model import AnomalyDetectionModel

class OutlierDetectionModel(AnomalyDetectionModel):

    def __init__(self, threshold=3):
        self.threshold = threshold
        self.load_model()

    def load_model(self):
        pass        # there is nothing to load

    def detect(self, data):
        velocity = data.get('velocity', None)
        if velocity is None:
            return False, {}

        # Simple z-score based outlier detection
        # For demonstration, assume mean=5, std=1.5
        mean = 5
        std = 1.5
        z_score = (velocity - mean) / std

        if abs(z_score) > self.threshold:
            anomaly_details = {
                'metric': 'velocity',
                'value': velocity,
                'z_score': z_score
            }
            return True, anomaly_details
        return False, {}
