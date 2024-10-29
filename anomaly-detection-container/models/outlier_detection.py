import numpy as np
from anomaly_detection_model import AnomalyDetectionModel

class OutlierDetectionModel(AnomalyDetectionModel):
    """
    Simple outlier detection based on statistical thresholds.
    """

    def __init__(self, threshold=3):
        self.threshold = threshold
        self.load_model()

    def load_model(self):
        """
        For this simple model, there's no pre-trained model to load.
        """
        pass

    def detect(self, data):
        velocity = data.get('velocity', None)
        if velocity is None:
            return False, {}
        
        # Simple z-score based outlier detection
        mean = 5
        std = 1.5
        z_score = (velocity - mean) / std

        if abs(z_score) > self.threshold:
            anomaly_details = {
                'metric': 'velocity',
                'value': velocity,
                'z_score': z_score,
                'anomaly_type': 'high_velocity',
                'message': f'Velocity outlier detected with z-score {z_score:.2f}'
            }
            return True, anomaly_details
        return False, {}
