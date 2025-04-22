
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

class ConstellationAnomalyDetectionModel(ABC):
    """
    Abstract base class for constellation-level anomaly detection models.
    """

    @abstractmethod
    def detect(self, time, satellite_id, data) -> tuple[bool, list[AnomalyDetails] | None]:
        """
        Process incoming data and detect anomalies at the constellation level.

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
        Load or initialize the model. This method can be used to load pre-trained models.
        """
        pass

    @abstractmethod
    def save_model(self, model_path: str):
        """Persist model weights or parameters."""
        pass
