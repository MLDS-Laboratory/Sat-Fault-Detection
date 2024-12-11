
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
    def detect(self, data) -> tuple[bool, AnomalyDetails]:
        """
        Process incoming data and detect anomalies at the constellation level.

        Parameters:
            data (dict): A dictionary containing telemetry data. Data format:
            {
                'time': number,
                'satellite_id': number,
                'data': dict
            }
            Since data comes in one satellite at a time, this class will need to track the satellites together.

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
