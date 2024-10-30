import os
import sys
import json
import importlib.util
import logging
from kafka import KafkaConsumer
from influxdb import InfluxDBClient
from models.constellation.constellation_anomaly_detection_model import ConstellationAnomalyDetectionModel
from models.satellite.satellite_anomaly_detection_model import SatelliteAnomalyDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Path to the models directory and config file
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

class AnomalyDetectionManager:
    def __init__(self, influx_host='influx_telegraf', influx_port=8086, influx_db='telemetry_db', config_path=CONFIG_FILE):
        self.constellation_models = []
        self.satellite_models = {}
        self.load_models(config_path)
        self.influx_client = InfluxDBClient(host=influx_host, port=influx_port, database=influx_db)
        self.ensure_anomalies_measurement()

    def load_models(self, config_path):
        """
        Dynamically load all active anomaly detection models from the config file.
        """
        if not os.path.exists(config_path):
            logging.error(f"Configuration file {config_path} not found.")
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load constellation-level models
        constellation_models = config.get('constellation_models', {})
        for model_name, model_file in constellation_models.items():
            model_path = os.path.join(MODELS_DIR, model_file)
            if not os.path.exists(model_path):
                logging.warning(f"Model file {model_file} not found in {MODELS_DIR}. Skipping.")
                continue

            spec = importlib.util.spec_from_file_location(model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Instantiate the model
            model_class = getattr(module, model_name, None)
            if model_class and issubclass(model_class, ConstellationAnomalyDetectionModel):
                model_instance = model_class()
                model_instance.load_model()
                self.constellation_models.append(model_instance)
                logging.info(f"Loaded constellation-level model: {model_name}")
            else:
                logging.warning(f"Model class {model_name} not found or does not inherit from ConstellationAnomalyDetectionModel.")

        # Load satellite-specific models
        satellite_models = config.get('satellite_models', {})
        for model_name, model_file in satellite_models.items():
            model_path = os.path.join(MODELS_DIR, model_file)
            if not os.path.exists(model_path):
                logging.warning(f"Model file {model_file} not found in {MODELS_DIR}. Skipping.")
                continue

            spec = importlib.util.spec_from_file_location(model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Instantiate the model
            model_class = getattr(module, model_name, None)
            if model_class and issubclass(model_class, SatelliteAnomalyDetectionModel):
                model_instance = model_class()
                model_instance.load_model()
                self.satellite_models[model_name] = model_instance
                logging.info(f"Loaded satellite-specific model: {model_name}")
            else:
                logging.warning(f"Model class {model_name} not found or does not inherit from SatelliteAnomalyDetectionModel.")

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
            data (dict): A dictionary containing telemetry data.
        """
        
        # Process constellation-level anomalies
        for model in self.constellation_models:
            try:
                is_anomaly, details = model.detect(data)
                if is_anomaly:
                    self.record_anomaly(details)
            except Exception as e:
                logging.error(f"Error in constellation model {model.__class__.__name__}: {e}")

        # Process satellite-specific anomalies
        satellite_data = {}
        for record in data:
            sat_id = record.get('satellite_id')
            if sat_id is None:
                continue
            if sat_id not in satellite_data:
                satellite_data[sat_id] = []
            satellite_data[sat_id].append(record)

        for sat_id, records in satellite_data.items():
            for model_name, model in self.satellite_models.items():
                for record in records:
                    try:
                        is_anomaly, details = model.detect(record)
                        if is_anomaly:
                            self.record_anomaly(details)
                    except Exception as e:
                        logging.error(f"Error in satellite model {model_name} for satellite {sat_id}: {e}")

    def record_anomaly(self, details):
        """
        Record the anomaly in InfluxDB.

        Parameters:
            details (dict): Details about the detected anomaly.
        """
        anomaly = {
            "measurement": "anomalies",
            "tags": {
                "satellite_id": str(details.get('satellite_id', 'unknown')),
                "anomaly_model": details.get('anomaly_model', 'unknown')
            },
            "time": details.get('time'),
            "fields": {
                "metric": details.get('metric', 'unknown'),
                "anomalous_value": details.get('value', 0),
                "message": details.get('message', '')
            }
        }
        self.influx_client.write_points([anomaly])
        logging.info(f"Anomaly detected and recorded: {anomaly}")

def main():
    # Initialize the anomaly detection manager
    manager = AnomalyDetectionManager()

    # Initialize Kafka Consumer
    consumer = KafkaConsumer(
        'telemetry',
        bootstrap_servers='kafka:9092',
        api_version=(0, 10, 1),
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='anomaly_detection_group'
    )

    logging.info("Anomaly Detection Service Started. Listening for telemetry data...")

    for message in consumer:
        telemetry_data = message.value
        manager.process_data(telemetry_data)

if __name__ == "__main__":
    main()
