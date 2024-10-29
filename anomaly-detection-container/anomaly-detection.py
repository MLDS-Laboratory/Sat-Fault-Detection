import os
import sys
import json
import importlib.util
from kafka import KafkaConsumer
from influxdb import InfluxDBClient

# Path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

class AnomalyDetectionManager:
    
    def __init__(self, influx_host='influxdb', influx_port=8086, influx_db='telemetry_db'):
        self.models = []
        self.load_models()
        self.influx_client = InfluxDBClient(host=influx_host, port=influx_port, database=influx_db)
        self.ensure_anomalies_measurement()

    def load_models(self):
        """
        Dynamically load all anomaly detection models from the models directory.
        """
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.py') and filename != 'anomaly_detection_model.py':
                module_name = filename[:-3]
                module_path = os.path.join(MODELS_DIR, filename)
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find the class that inherits from AnomalyDetectionModel
                for attr in dir(module):
                    cls = getattr(module, attr)
                    if isinstance(cls, type) and issubclass(cls, module.anomaly_detection_model.AnomalyDetectionModel) and cls is not module.anomaly_detection_model.AnomalyDetectionModel:
                        self.models.append(cls())
                        print(f"Loaded model: {cls.__name__}")

    def ensure_anomalies_measurement(self):
        """
        Ensure that the 'anomalies' measurement exists in InfluxDB.
        """
        self.influx_client.switch_database('telemetry_db')
        # InfluxDB creates measurements on first write, so this might be optional

    def process_data(self, data):
        """
        Pass the data to each model to detect anomalies.

        Parameters:
            data (dict): The telemetry data.

        """
        for model in self.models:
            is_anomaly, details = model.detect(data)
            if is_anomaly:
                self.record_anomaly(data, details)

    def record_anomaly(self, data, details):
        """
        Record the anomaly in InfluxDB.

        Parameters:
            data (dict): The original telemetry data.
            details (dict): Details about the detected anomaly.
        """
        anomaly = {
            "measurement": "anomalies",
            "tags": {
                "satellite_id": str(data.get('satellite_id', 'unknown')),
                "anomaly_type": details.get('anomaly_type', 'unknown')
            },
            "time": data.get('time'),
            "fields": {
                "metric": details.get('metric', 'unknown'),
                "value": details.get('value', 0),
                "additional_info": json.dumps(details)
            }
        }
        self.influx_client.write_points([anomaly])
        print(f"Anomaly detected and recorded: {anomaly}")

def main():
    # Initialize the anomaly detection manager
    manager = AnomalyDetectionManager()

    # Initialize Kafka Consumer
    consumer = KafkaConsumer(
        'telemetry',
        bootstrap_servers='kafka:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='anomaly_detection_group'
    )

    print("Anomaly Detection Service Started. Listening for telemetry data...")

    for message in consumer:
        telemetry_data = message.value
        manager.process_data(telemetry_data)

if __name__ == "__main__":
    main()
