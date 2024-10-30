from abc import ABC, abstractmethod

class SatelliteAnomalyDetectionModel(ABC):
    """
    Abstract base class for satellite-specific anomaly detection models.
    """

    @abstractmethod
    def detect(self, data):
        """
        Process incoming data for a specific satellite and detect anomalies.

        Parameters:
            data (dict): A dictionary containing telemetry data for a single satellite.

        Returns:
            bool: True if anomaly is detected, False otherwise.
            dict: Details of the anomaly if detected.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Load or initialize the satellite-specific model. This method can be used to load pre-trained models.
        """
        pass
