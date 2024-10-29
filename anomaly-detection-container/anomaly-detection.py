from kafka import KafkaConsumer, KafkaProducer
import json

def detect_anomalies():
    consumer = KafkaConsumer(
        'telemetry',
        bootstrap_servers='kafka:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        group_id='anomaly_detection_group'
    )
    producer = KafkaProducer(
        bootstrap_servers='kafka:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    for message in consumer:
        data = message.value
        # Simple outlier detection
        telemetry_data = data['data']
        if 'velocity' in telemetry_data and telemetry_data['velocity'] > 9.5:
            anomaly = {
                'time': data['time'],
                'satellite_id': data['satellite_id'],
                'metric': 'velocity',
                'value': telemetry_data['velocity'],
                'anomaly_type': 'high_velocity'
            }
            producer.send('anomalies', value=anomaly)
    producer.flush()

if __name__ == "__main__":
    detect_anomalies()
