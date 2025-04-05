from models.constellation.constellation_anomaly_detection_model import ConstellationAnomalyDetectionModel, AnomalyDetails
import numpy as np

class ConstellationOutlierDetection(ConstellationAnomalyDetectionModel):
    """
    Example constellation-level anomaly detection model using statistical outliers over a sliding time window.
    """

    def __init__(self, std_threshold=2, window_seconds=30):
        """
        Parameters:
            std_threshold (float): Threshold (in multiples of standard deviation) for detecting outliers.
            window_seconds (int): Duration (in seconds) of the sliding window over which to calculate statistics.
        """
        self.std_threshold = std_threshold
        self.window = window_seconds
        
        # A dictionary to store history of readings for each channel.
        # Keys are channel names; values are lists of tuples (time, value).
        self.channel_history = {}
        self.load_model()

    def load_model(self):
        """
        Initialize any required parameters or load pre-trained models.
        For a simple statistical model, no initialization is needed.
        """
        pass

    def detect(self, time, satellite_id, data) -> tuple[bool, list[AnomalyDetails]]:
        """
        Detect anomalies on a per-satellite basis by comparing each channel's current reading against
        the running mean and standard deviation computed over a sliding window across all satellites.

        Parameters:
            time (int): The time of the data point.
            satellite_id (int): The ID of the satellite.
            data (dict): A dictionary containing telemetry data channels and their values.

        Returns:
            tuple[bool, list[AnomalyDetails]]:
                - bool: True if at least one anomaly is detected; False otherwise.
                - list[AnomalyDetails]: List of anomaly details instances.
        """
        anomalies = []

        # Update internal history for each channel in the current data
        for channel, value in data.items():
            # Ensure the value is numeric before updating history
            if not isinstance(value, (int, float)):
                continue
            if channel not in self.channel_history:
                self.channel_history[channel] = []
            self.channel_history[channel].append((time, value))

        # Prune old entries: keep only records within the sliding window
        for channel in self.channel_history:
            self.channel_history[channel] = [(t, v) for t, v in self.channel_history[channel] if t >= time - self.window]

        # Check for anomalies in each channel in the current satellite data
        for channel, current_value in data.items():
            if not isinstance(current_value, (int, float)):
                continue

            history = self.channel_history.get(channel, [])
            # Only consider channels with at least two data points for statistics calculation
            if len(history) < 2:
                continue

            values = [v for t, v in history]
            mean_val = np.mean(values)
            std_val = np.std(values)

            # Avoid division by zero and ignore cases with no variation
            if std_val == 0:
                continue

            # If the current reading is outside of the confidence bounds, flag an anomaly
            if abs(current_value - mean_val) > self.std_threshold * std_val:
                anomaly = AnomalyDetails(
                    satellite_id=satellite_id,
                    anomaly_model=self.__class__.__name__,
                    time=time,
                    time_end=time,  # In this simple case, time_end is set equal to time.
                    metric=channel,
                    value=current_value,
                    message=f"{channel} value {current_value} deviates from mean {mean_val:.2f} by more than "
                            f"{self.std_threshold} standard deviations ({std_val:.2f})."
                )
                anomalies.append(anomaly)

        if anomalies:
            return True, anomalies
        return False, []
