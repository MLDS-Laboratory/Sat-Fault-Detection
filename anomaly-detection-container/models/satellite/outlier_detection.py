from models.satellite.satellite_anomaly_detection_model import SatelliteAnomalyDetectionModel       # imports relative to docker container root
import logging

class OutlierDetectionModel(SatelliteAnomalyDetectionModel):
    """
    Simple outlier detection based on statistical thresholds.
    """

    def __init__(self, metric = None, threshold=3):
        """
        Parameters:
            threshold (float): Z-score threshold for outlier detection.
            metric (str): The data field to use for outlier detection.
                        If None, the model will check each data field for outliers.
        """

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
        
        logging.info(f'Velocity: {velocity}')
        
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
