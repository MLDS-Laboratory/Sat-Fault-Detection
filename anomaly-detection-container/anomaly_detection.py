import os
import sys
import json
import importlib.util
import logging
from kafka import KafkaConsumer, KafkaProducer
from influxdb import InfluxDBClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Path to the models directory and config file
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

class AnomalyDetectionManager:
    def __init__(self, influx_host='influx_telegraf', influx_port=8086, influx_db='telemetry_db', config_path=CONFIG_FILE):
        self.models = []
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

        active_models = config.get('active_models', {})
        for model_name in active_models:

            model_file = str(active_models[model_name])
            model_path = os.path.join(MODELS_DIR, model_file)
            if not os.path.exists(model_path):
                logging.warning(f"Model file {model_file} not found in {MODELS_DIR}. Skipping.")
                continue

            spec = importlib.util.spec_from_file_location(model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Instantiate the model
            model_class = getattr(module, model_name, None)
            if model_class and issubclass(model_class, module.AnomalyDetectionModel):
                self.models.append(model_class())
                logging.info(f"Loaded model: {model_name}")
            else:
                logging.warning(f"Model class {model_name} not found or does not inherit from AnomalyDetectionModel.")

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
            try:
                is_anomaly, details = model.detect(data)
                if is_anomaly:
                    self.record_anomaly(data, details)
            except Exception as e:
                logging.error(f"Error in model {model.__class__.__name__}: {e}")

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
                "anomaly_model": details.get('anomaly_type', 'unknown')
            },
            "time": data.get('time'),
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
