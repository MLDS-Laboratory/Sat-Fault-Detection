import random
import time
import psycopg2
from psycopg2 import sql
import json  # Ensure you import json for telemetry data

def generate_data(simulation_id, is_constellation=False, satellites=1):
    # Remove the password field and use 127.0.0.1 instead of localhost
    conn = psycopg2.connect(
        host="127.0.0.1",  # Use IPv4 loopback to avoid any IPv6 issues
        database="telemetry_db",
        user="postgres",
    )
    cursor = conn.cursor()

    # Create a new table with flexible schema
    table_name = f"simulation_{simulation_id}"
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            time INTEGER,
            satellite_id INTEGER,
            data JSONB,
            PRIMARY KEY (time, satellite_id)
        );
    """).format(sql.Identifier(table_name)))
    conn.commit()

    # Simulate data generation
    for t in range(100):  # Simulate 100 timesteps
        for sat_id in range(1, satellites + 1):
            telemetry_data = {
                "position": random.uniform(0, 1000),
                "velocity": random.uniform(0, 10),
                # Add more fields as needed
            }
            cursor.execute(sql.SQL("""
                INSERT INTO {} (time, satellite_id, data)
                VALUES (%s, %s, %s);
            """).format(sql.Identifier(table_name)), (t, sat_id, json.dumps(telemetry_data)))
        conn.commit()
        time.sleep(0.1)  # Simulate real-time intervals

    cursor.close()
    conn.close()

if __name__ == "__main__":
    generate_data(simulation_id=random.uniform(0, 20000), is_constellation=True, satellites=3)
