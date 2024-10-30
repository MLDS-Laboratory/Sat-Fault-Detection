from constellation_anomaly_detection_model import ConstellationAnomalyDetectionModel
import numpy as np

class ConstellationOutlierDetection(ConstellationAnomalyDetectionModel):
    """
    Example constellation-level anomaly detection model using statistical outliers.
    """

    def load_model(self):
        """
        Initialize any required parameters or load pre-trained models.
        """
        # For a simple statistical model, no initialization is needed
        pass

    def detect(self, data):
        """
        Detect anomalies based on the average velocity of the constellation.
        """
        velocities = [sat['data']['velocity'] for sat in data if 'velocity' in sat['data']]
        if not velocities:
            return False, {}

        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)

        anomalies = []
        for sat in data:
            velocity = sat['data'].get('velocity')
            if velocity is not None and abs(velocity - mean_velocity) > 2 * std_velocity:
                anomalies.append({
                    'satellite_id': sat['satellite_id'],
                    'metric': 'velocity',
                    'value': velocity,
                    'message': 'Velocity deviates significantly from constellation average.'
                })

        if anomalies:
            return True, {'anomalies': anomalies}
        return False, {}
