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
        self.metric = metric
        self.load_model()

    def load_model(self):
        """
        For this simple model, there's no pre-trained model to load.
        """
        pass

    def detect(self, data):
        """
        Detect outliers in the incoming data.

        Parameters:
            data (dict): A dictionary containing telemetry data.

        Returns:
            bool: True if an anomaly is detected, False otherwise.
            dict: Details of the anomaly if detected.
        """
        if self.metric is not None:
            return self.detect_field(self.metric, data)
        else:
            for field in data:
                is_anomaly, details = self.detect_field(field, data)
                if is_anomaly:
                    return is_anomaly, details
        return False, {}

    def detect_field(self, field, data):
        field = data.get(field, None)
        if field is None:
            return False, {}
        
        logging.info(f'Velocity: {field}')
        
        # Simple z-score based outlier detection
        mean = 5
        std = 1.5
        z_score = (field - mean) / std

        if abs(z_score) > self.threshold:
            anomaly_details = {
                'metric': 'field',
                'value': field,
                'z_score': z_score,
                'anomaly_type': 'high_field',
                'message': f'Velocity outlier detected with z-score {z_score:.2f}'
            }
            return True, anomaly_details
        return False, {}
