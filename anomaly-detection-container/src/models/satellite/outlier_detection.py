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
        self.means = {}
        self.variances = {}
        self.counts = {}
        self.anomaly_start = {}

    def load_model(self):
        """
        For this simple model, there's no pre-trained model to load.
        """
        pass

    def detect(self, time, satellite_id, data):
        """
        Detect outliers in the incoming data.

        Parameters:
            time (int): The timestamp of the data point.
            satellite_id (int): The ID of the satellite.
            data (dict): A dictionary containing telemetry data fields and their values.

        Returns:
            bool: True if an anomaly is detected, False otherwise.
            list[AnomalyDetails]: Details of the anomaly (or multiple) if detected. Otherwise, return None.
        """
        logging.info(f'Outlier detection model reports: {self.metric} - {self.threshold}')
        self.satellite_id = satellite_id  # Store satellite_id for use in detect_field
        
        if self.metric is not None and not isinstance(self.metric, list):
            anomaly_data = self.detect_field(time, self.metric, data)
            if anomaly_data is not None:
                return True, anomaly_data
            else:
                return False, None
            
        elif isinstance(self.metric, list):
            for field in self.metric:
                logging.info(f"--------------------------------- {field}")
                anomaly_data = self.detect_field(time, field, data)
                if anomaly_data is not None:
                    return True, anomaly_data
        else:
            anomalies = []
            for field in data:
                anomaly_data = self.detect_field(time, field, data)
                if anomaly_data is not None:
                    anomalies.append(anomaly_data)
            
            if anomalies:
                return True, anomalies
                
        return False, None

    def detect_field(self, time, field, data):
        """
        Detect outliers in a specific field of the incoming data.

        Parameters:
            time (int): The timestamp of the data point.
            field (str): The field to check for outliers.
            data (dict): A dictionary containing the payload fields and their values.

        Returns:
            AnomalyDetails: Details of the anomaly if detected. Otherwise, return None
        """
        field_name = field
        field_value = data.get(field, None)
        logging.info(f'Analyzing field: {field_name} - Value: {field_value}')
        
        if field_value is None:
            return None
        
        # Skip non-numeric values
        if not isinstance(field_value, (int, float)):
            return None
        
        # Initialize tracking for this field if it's the first time we see it
        if field_name not in self.means:
            self.means[field_name] = 0
            self.variances[field_name] = 0
            self.counts[field_name] = 0
            self.anomaly_start[field_name] = None

        # Update running statistics using Welford's online algorithm
        self.means[field_name] = (self.means[field_name] * self.counts[field_name] + field_value) / (self.counts[field_name] + 1)
        self.variances[field_name] = (self.variances[field_name] * (self.counts[field_name]) + 
                                     (field_value - self.means[field_name]) ** 2) / (self.counts[field_name] + 1)  
        self.counts[field_name] += 1

        mean = self.means[field_name]
        std = self.variances[field_name] ** 0.5
        
        # Avoid division by zero
        if std == 0:
            return None

        # Calculate z-score
        z_score = (field_value - mean) / std

        # if it is an anomaly
        if abs(z_score) > self.threshold:
            # Check if anomaly already started
            if field_name not in self.anomaly_start or self.anomaly_start[field_name] is None:
                self.anomaly_start[field_name] = time
                ending_time = time
            else:
                ending_time = time
                time = self.anomaly_start[field_name]

            anomaly_details = AnomalyDetails(
                    satellite_id=self.satellite_id, 
                    anomaly_model=self.__class__.__name__, 
                    time=time,
                    metric=field_name, 
                    value=field_value, 
                    time_end=ending_time,
                    message=f'{field_name} outlier detected with z-score {z_score:.2f}'
                )
            
            return anomaly_details
        else:
            self.anomaly_start[field_name] = None  # Reset the anomaly start time
        
        return None
    

    def save_model(self, model_path: str):
        pass
