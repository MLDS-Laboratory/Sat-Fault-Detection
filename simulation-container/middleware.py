import random
import time
import psycopg2
from psycopg2 import sql
import json
import threading
from kafka import KafkaProducer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Middleware Function
def stream_data(table_name, kafka_topic='telemetry', satellite_id=None, speed = 1):
    conn = psycopg2.connect(
        host="localhost",
        database="telemetry_db",
        user="postgres",
        password="postgres"
    )
    cursor = conn.cursor()
    producer = KafkaProducer(
        bootstrap_servers='kafka:9092',
        api_version=(0, 10, 1),
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    query = sql.SQL("SELECT time, satellite_id, data FROM {}").format(sql.Identifier(table_name))
    if satellite_id:
        query += sql.SQL(" WHERE satellite_id = %s")
        cursor.execute(query, (satellite_id,))
    else:
        cursor.execute(query)

    rows = cursor.fetchall()
    for i, row in enumerate(rows):
        message = {
            "time": row[0],
            "satellite_id": row[1],
            "data": row[2]
        }
        producer.send(kafka_topic, value=message)
        if i < len(rows) - 1:
            next_time = rows[i + 1][0]
            current_time = row[0]
            time.sleep(max(0, next_time - current_time) / speed)
    producer.flush()
    logging.info("Data streaming completed.")

if __name__ == "__main__":

    time.sleep(2)  # Make sure to wait for InfluxDB and Kafka to start

    # Simulation Parameters
    table_name = "simulation_2"
    is_constellation = True
    satellites = 2
    speed = 50  # Speed up the simulation by __ times

    # Stream Data
    if is_constellation:
        threads = []
        for sat_id in range(1, satellites + 1):
            t = threading.Thread(target=stream_data, args=(table_name, 'telemetry', sat_id, speed))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        stream_data(table_name, speed=speed)
