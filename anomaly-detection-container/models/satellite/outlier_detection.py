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
            data = self.detect_field(data['time'], self.metric, data['data'])
            if data is not None:
                return True, data
            else:
                return False, None
        else:
            for field in data:
                anomaly_data = self.detect_field(data['time'], field, data['data'])
                if anomaly_data is not None:
                    return True, anomaly_data
        return False, None

    def detect_field(self, time, field, data):
        """
        Detect outliers in a specific field of the incoming data.

        Parameters:
            field (str): The field to check for outliers.
            data (dict): A dictionary containing the payload. This is the 'data' field in the whole telemetry packet.

        Returns:
            AnomalyDetails: Details of the anomaly if detected. Otherwise, return None
        """
        field_name = field
        field = data.get(field, None)
        logging.info(f'YEEEEEEEEEEEEEEE: {field} - {self.means} - {self.variances} - {self.counts}')
        if field is None:
            return None
        
        # keep running mean and std for the field
        if field_name not in self.means:
            self.means[field_name] = 0
            self.variances[field_name] = 0
            self.counts[field_name] = 0

        self.means[field_name] = (self.means[field_name] * self.counts[field_name] + field) / (self.counts[field_name] + 1)
        # Welford's online algorithm for variance
        self.variances[field_name] = (self.variances[field_name] * (self.counts[field_name]) + (field - self.means[field_name]) ** 2) / (self.counts[field_name] + 1)  
        self.counts[field_name] += 1

        mean = self.means[field_name]
        std = self.variances[field_name] ** 0.5

        # Simple z-score based outlier detection
        z_score = (field - mean) / std

        # if it is an anomaly
        if abs(z_score) > self.threshold:

            # check if anomaly already started
            if self.anomaly_start[field_name] is None:
                self.anomaly_start[field_name] = time
                ending_time = time
            else:
                ending_time = time
                time = self.anomaly_start[field_name]
                

            anomaly_details = AnomalyDetails(
                    satellite_id=self.satellite_id, anomaly_model=self.__class__.__name__, time=time,
                    metric=field_name, value=field, time_end=ending_time,
                    message=f'{field_name} outlier detected with z-score {z_score:.2f}'
                )
            
            return anomaly_details
        else:
            self.anomaly_start[field_name] = None       # reset the anomaly start time
        return None
