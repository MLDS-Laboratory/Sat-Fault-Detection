from models.constellation.constellation_anomaly_detection_model import ConstellationAnomalyDetectionModel
import numpy as np

class ConstellationOutlierDetection(ConstellationAnomalyDetectionModel):
    """
    Example constellation-level anomaly detection model using statistical outliers.
    """

    def __init__(self, std_threshold=2):
        """
        Parameters:
            std_threshold (float): The threshold for detecting outliers based on standard deviation.
        """
        self.std_threshold = std_threshold
        self.load_model()

    def load_model(self):
        """
        Initialize any required parameters or load pre-trained models.
        """
        # For a simple statistical model, no initialization is needed
        pass

    def detect(self, data):
        """
        Detect anomalies based on the average velocity of the constellation.

        data (dict): A dictionary containing telemetry data for all satellites. Data format:
            {
                'time': number,
                'satellite_id': number,
                'data': dict
            }
            Since data comes in one satellite at a time, this will need to track the satellites together
        """

        # if not a list, make it one
        if not isinstance(data, list):
            data = [data]

        velocities = [sat['data']['velocity'] for sat in data if 'velocity' in sat['data']]
        if not velocities:
            return False, {}

        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)

        anomalies = []
        for sat in data:
            velocity = sat['data'].get('velocity')
            if velocity is not None and abs(velocity - mean_velocity) > self.std_threshold * std_velocity:
                anomalies.append({
                    'satellite_id': sat['satellite_id'],
                    'metric': 'velocity',
                    'value': velocity,
                    'message': 'Velocity deviates significantly from constellation average.'
                })

        if anomalies:
            return True, {'anomalies': anomalies}
        return False, {}
