from abc import ABC, abstractmethod
from dataclasses import dataclass

# Details class for an anomaly detection model to return
@dataclass
class AnomalyDetails:
    satellite_id: int
    anomaly_model: str
    time: int
    time_end: int
    metric: str
    value: float
    message: str

class SatelliteAnomalyDetectionModel(ABC):
    """
    Abstract base class for satellite-specific anomaly detection models.
    """
    def __init__(self, **kwargs):
        """
        Initialize the satellite-specific anomaly detection model. 

        Parameters:
            satellite_id (int): The ID of the satellite.
        """
        self.satellite_id = kwargs.get('satellite_id')
        self.load_model()

    @abstractmethod
    def detect(self, data) -> AnomalyDetails:
        """
        Process incoming data for a specific satellite and detect anomalies.

        Parameters:
            data (dict): A dictionary containing telemetry data for a single satellite.

        Returns:
            AnomalyDetails: Details of the anomaly if detected. Otherwise, return None.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Load or initialize the satellite-specific model. This method can be used to load pre-trained models.
        """
        pass
