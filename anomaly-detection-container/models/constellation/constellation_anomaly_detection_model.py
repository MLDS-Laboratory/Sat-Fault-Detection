
from abc import ABC, abstractmethod

class ConstellationAnomalyDetectionModel(ABC):
    """
    Abstract base class for constellation-level anomaly detection models.
    """

    @abstractmethod
    def detect(self, data):
        """
        Process incoming data and detect anomalies at the constellation level.

        Parameters:
            data (dict): A dictionary containing telemetry data.

        Returns:
            bool: True if anomaly is detected, False otherwise.
            dict: Details of the anomaly if detected.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Load or initialize the model. This method can be used to load pre-trained models.
        """
        pass
