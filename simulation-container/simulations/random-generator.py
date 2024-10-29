import random
import time
import psycopg2
from psycopg2 import sql
import json  

def generate_data(simulation_id, is_constellation=False, satellites=1):
    conn = psycopg2.connect(
        host="localhost",
        database="telemetry_db",
        user="postgres",
    )
    cursor = conn.cursor()

    # Create a new table with flexible schema (JSONB for now, but later maybe column headers)
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

        print('Data generated for timestep', t)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    generate_data(simulation_id=3, is_constellation=True, satellites=3)

    # print the table names
    conn = psycopg2.connect(
        host="localhost",
        database="telemetry_db",
        user="postgres",
    )
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")

    for table in cursor.fetchall():
        print(table)

    cursor.close()
    conn.close()
