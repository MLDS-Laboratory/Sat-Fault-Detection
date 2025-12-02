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
    def detect(self, time, satellite_id, data) -> tuple[bool, list[AnomalyDetails] | None]:
        """
        Process incoming data for a specific satellite and detect anomalies.

        Parameters:
            time (int): The time of the data point.
            satellite_id (int): The ID of the satellite.
            data (dict): A dictionary containing the telemetry data from any channels at this timestep.

        Returns:
            bool: True if anomaly is detected, False otherwise.
            list[AnomalyDetails]: Details of the anomaly (or multiple anomalies) if detected. Otherwise, return None.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Load or initialize the satellite-specific model. This method can be used to load pre-trained models.
        """
        pass

    @abstractmethod
    def save_model(self, model_path: str):
        """Persist model weights or parameters."""
        pass
