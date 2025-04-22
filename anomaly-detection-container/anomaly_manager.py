import os
import sys
import json
import importlib.util
import logging
import traceback
from kafka import KafkaConsumer
from influxdb import InfluxDBClient
from models.constellation.constellation_anomaly_detection_model import ConstellationAnomalyDetectionModel
from models.satellite.satellite_anomaly_detection_model import SatelliteAnomalyDetectionModel, AnomalyDetails

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Path to the models directory and config file
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

class AnomalyDetectionManager:
    def __init__(self, influx_host='influx_telegraf', influx_port=8086, influx_db='telemetry_db', config_path=CONFIG_FILE):
        self.constellation_models = []
        self.satellite_models_classes = {}
        self.active_satellite_models = {}
        self.seen_satellite_ids = set()
        self.load_models(config_path)
        self.influx_client = InfluxDBClient(host=influx_host, port=influx_port, database=influx_db)
        self.ensure_anomalies_measurement()

    def load_models(self, config_path):
        """
        Dynamically load all active anomaly detection models from the config file with parameters.
        """
        if not os.path.exists(config_path):
            logging.error(f"Configuration file {config_path} not found.")
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Load constellation-level models
        constellation_models = config.get('constellation_models', {})
        for model_name, model_info in constellation_models.items():
            model_path = os.path.join(MODELS_DIR, model_info['path'])
            parameters = model_info.get('parameters', {})

            self._load_constellation_model(
                model_name, model_path, parameters
            )

        # Load satellite-specific models
        satellite_models = config.get('satellite_models', {})
        for model_name, model_info in satellite_models.items():
            model_path = os.path.join(MODELS_DIR, model_info['path'])
            parameters = model_info.get('parameters', {})

            self._register_satellite_model(
                model_name, model_path, parameters
            )

    def _load_constellation_model(self, model_name, model_path, parameters):
        """
        Helper function to load a constellation-level model dynamically with parameters.
        """
        if not os.path.exists(model_path):
            logging.warning(f"Constellation model file {model_path} not found. Skipping.")
            return

        spec = importlib.util.spec_from_file_location(model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Instantiate the model with parameters as kwargs
        model_class = getattr(module, model_name, None)
        if model_class and issubclass(model_class, ConstellationAnomalyDetectionModel):
            try:
                model_instance = model_class(**parameters)
                model_instance.load_model()
                self.constellation_models.append(model_instance)
                logging.info(f"Loaded constellation-level model: {model_name} with parameters: {parameters}")
            except Exception as e:
                logging.error(f"Error initializing constellation model {model_name}: {e}")
        else:
            logging.warning(f"Model class {model_name} not found or does not inherit from ConstellationAnomalyDetectionModel.")

    def _register_satellite_model(self, model_name, model_path, parameters):
        """
        Register satellite-specific model classes and parameters without instantiating them yet.
        """
        if not os.path.exists(model_path):
            logging.warning(f"Satellite model file {model_path} not found. Skipping.")
            return

        spec = importlib.util.spec_from_file_location(model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the model class
        model_class = getattr(module, model_name, None)
        if model_class and issubclass(model_class, SatelliteAnomalyDetectionModel):
            # Store the class and parameters for later instantiation
            self.satellite_models_classes[model_name] = {
                'class': model_class,
                'parameters': parameters
            }
            logging.info(f"Registered satellite-specific model class: {model_name} with parameters: {parameters}")
        else:
            logging.warning(f"Satellite model class {model_name} not found or does not inherit from SatelliteAnomalyDetectionModel.")

    def ensure_anomalies_measurement(self):
        """
        Ensure that the 'anomalies' measurement exists in InfluxDB.
        """
        self.influx_client.switch_database('telemetry_db')

    def process_data(self, data):
        """
        Pass the data to each model to detect anomalies.

        Parameters:
            data (dict): A dictionary containing telemetry data.
        """
        satellite_id = int(data.get('satellite_id'))
        if satellite_id is None:
            logging.warning("Data record missing 'satellite_id'. Skipping satellite-specific anomaly detection.")
        else:
            # Check if this is a new satellite
            if satellite_id not in self.seen_satellite_ids:
                self.seen_satellite_ids.add(satellite_id)
                self.active_satellite_models[satellite_id] = {}
                # Instantiate satellite-specific models for this satellite
                for model_name, model_info in self.satellite_models_classes.items():
                    model_class = model_info['class']
                    parameters = model_info['parameters']
                    # Add the satellite_id to the parameters
                    parameters['satellite_id'] = satellite_id
                    try:
                        model_instance = model_class(**parameters)
                        model_instance.load_model()
                        self.active_satellite_models[satellite_id][model_name] = model_instance
                        logging.info(f"Loaded satellite-specific model: {model_name} for satellite {satellite_id} with parameters: {parameters}")
                    except Exception as e:
                        logging.error(f"Error initializing satellite model {model_name} for satellite {satellite_id}: {e}")

        # Process constellation-level anomalies
        for model in self.constellation_models:
            try:
                timestep = data.get('time', 0)
                this_satellite_id = data.get('satellite_id', 0)
                channel_data = data.get('data', {})
                is_anomaly, details = model.detect(timestep, this_satellite_id, channel_data)
                if is_anomaly:
                    # details could be a list of anomalies
                    if isinstance(details.get('anomalies'), list):
                        for anomaly_detail in details['anomalies']:
                            self.record_anomaly(anomaly_detail)
                    else:
                        self.record_anomaly(details)
            except Exception as e:
                logging.error(f"Error in constellation model {model.__class__.__name__}: {e} - {traceback.format_exc()}")

        # Process satellite-specific anomalies
        if satellite_id is not None and satellite_id in self.active_satellite_models:
            for model_name, model in self.active_satellite_models[satellite_id].items():
                try:
                    logging.info(f"Processing satellite-specific anomalies for satellite {satellite_id} with model {model_name}")
                    logging.info(f"Data: {data}")
                    timestep = data.get('time', 0)
                    channel_data = data.get('data', {})
                    satellite_id = data.get('satellite_id', 0)
                    is_anomaly, details = model.detect(timestep, satellite_id, channel_data)
                    if is_anomaly:
                        self.record_anomaly(details, True)
                except Exception as e:
                    logging.error(f"Error in satellite model {model_name} for satellite {satellite_id}: {e}")

    def record_anomaly(self, details, is_sat):
        """
        Record the anomaly in InfluxDB.

        Parameters:
            details (dict): Details about the detected anomaly.
            is_sat (bool): True if the anomaly is satellite-specific, False if constellation-level.
        """
        anomaly = {
            "measurement": "anomalies",
            "tags": {
                "satellite_id": str(details.satellite_id),
                "anomaly_model": details.anomaly_model,
                "is_sat": is_sat
            },
            "time": details.time,
            "fields": {
                "metric": details.metric,
                "anomalous_value": details.value,
                "message": details.message,
                "time_end": (details.time_end if (hasattr(details, 'time_end') and details.time_end is not None) else details.time)
            }
        }
        self.influx_client.write_points([anomaly], time_precision='s')
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
