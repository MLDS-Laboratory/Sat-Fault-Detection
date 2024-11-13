from models.satellite.satellite_anomaly_detection_model import SatelliteAnomalyDetectionModel, AnomalyDetails       # imports relative to docker container root
import logging

class OutlierDetectionModel(SatelliteAnomalyDetectionModel):
    """
    Simple outlier detection based on statistical thresholds.
    """

    def __init__(self, metric = None, threshold=3, **kwargs):
        """
        Parameters:
            threshold (float): Z-score threshold for outlier detection.
            metric (str): The data field to use for outlier detection.
                        If None, the model will check each data field for outliers.
        """
        super().__init__(**kwargs)
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
            data (dict): A dictionary containing telemetry data. Data format:
                {
                    'time': number,
                    'satellite_id': number,
                    'data': dict
                }

        Returns:
            bool: True if an anomaly is detected, False otherwise.
            AnomalyDetails: Details of the anomaly if detected. Otherwise, return None.
        """
        logging.info(f'Outlier detection model reports: {self.metric} - {self.threshold}')
        if self.metric is not None:
            return self.detect_field(data['time'], self.metric, data['data'])
        else:
            for field in data:
                is_anomaly, details = self.detect_field(data['time'], field, data['data'])
                if is_anomaly:
                    return details
        return None

    def detect_field(self, time, field, data):
        """
        Detect outliers in a specific field of the incoming data.

        Parameters:
            field (str): The field to check for outliers.
            data (dict): A dictionary containing the payload. This is the 'data' field in the whole telemetry packet.
        """
        field_name = field
        field = data.get(field, None)
        logging.info(f'Velocity: {field}')
        if field is None:
            return None
        
        
        # Simple z-score based outlier detection
        mean = 5
        std = 1.5
        z_score = (field - mean) / std

        if abs(z_score) > self.threshold:
            anomaly_details = AnomalyDetails(
                    satellite_id=self.satellite_id, anomaly_model=self.__class__.__name__, time=data['time'],
                    metric=field_name, value=field, 
                    message=f'Velocity outlier detected with z-score {z_score:.2f}'
                )
            
            return anomaly_details
        return None
