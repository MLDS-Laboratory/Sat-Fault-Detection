import psycopg2
from psycopg2 import sql
import json  
import frankensteinScenario as sc

def generate_data(simulation_id):
    conn = psycopg2.connect(
        host="localhost",
        database="telemetry_db",
        user="postgres",
    )
    cursor = conn.cursor()

    times, pos, velo, sigma, omega, CSSdata, disturbances, sensedSun, sunPoint = sc.run()
    # Create a new table with flexible schema (JSONB for now, but later maybe column headers)
    table_name = f"simulation_{simulation_id}"
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            time BIGINT PRIMARY KEY,
            x_pos DOUBLE PRECISION,
            y_pos DOUBLE PRECISION, 
            z_pos DOUBLE PRECISION,
            x_velo DOUBLE PRECISION,
            y_velo DOUBLE PRECISION
            z_velo DOUBLE PRECISION,
            x_sigma DOUBLE PRECISION, 
            y_sigma DOUBLE PRECISION, 
            z_sigma DOUBLE PRECISION,
            x_omega DOUBLE PRECISION,
            y_omega DOUBLE PRECISION,
            z_omega DOUBLE PRECISION,
            CSS_1 DOUBLE PRECISION,
            CSS_2 DOUBLE PRECISION,
            CSS_3 DOUBLE PRECISION,
            CSS_4 DOUBLE PRECISION,
            CSS_5 DOUBLE PRECISION,
            CSS_6 DOUBLE PRECISION,
            x_torque DOUBLE PRECISION,
            y_torque DOUBLE PRECISION,
            z_torque DOUBLE PRECISION,
            sun_x_sensed DOUBLE PRECISION,
            sun_y_sensed DOUBLE PRECISION,
            sun_z_sensed DOUBLE PRECISION,
            sun_x_true DOUBLE PRECISION,
            sun_y_true DOUBLE PRECISION,
            sun_z_true DOUBLE PRECISION
        );
    """).format(sql.Identifier(table_name)))
    conn.commit()

    for i in range(len(times)):
        cursor.execute(sql.SQL("""
                INSERT INTO {} (time, x_pos, 
                               y_pos, z_pos, 
                               x_velo, y_velo, 
                               z_velo, x_sigma, 
                               y_sigma, z_sigma, 
                               x_omega, y_omega, 
                               z_omega, CSS_1, 
                               CSS_2, CSS_3, 
                               CSS_4, CSS_5, 
                               CSS_6, x_torque, 
                               y_torque, z_torque, 
                               sun_x_sensed, sun_y_sensed, 
                               sun_z_sensed, sun_x_true, 
                               sun_y_true, sun_z_true)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 
                               %s, %s, %s, %s, %s, %s, %s, %s, 
                               %s, %s, %s, %s, %s, %s, %s, %s, 
                               %s, %s, %s, %s)
                               """)).format(sql.Identifier(table_name)), \
                                (times[i], pos[i, 0], pos[i, 1], pos[i, 2], \
                                 velo[i, 0], velo[i, 1], velo[i, 2], velo[i, 3], \
                                    sigma[i, 0], sigma[i, 1], sigma[i, 2], \
                                        sigma[i, 3], omega[i, 1], omega[i, 2], \
                                            omega[i, 3], CSSdata[0, i], CSSdata[1, i], \
                                                CSSdata[2, i], CSSdata[3, i], \
                                                    CSSdata[4, i], CSSdata[5, i], \
                                                        disturbances[i, 0], disturbances[i, 1], \
                                                            disturbances[i, 2], sensedSun[i, 0], \
                                                                sensedSun [i, 1], sensedSun[i, 2], \
                                                                    sunPoint[i, 0], sunPoint[i, 1], \
                                                                        sunPoint[i, 2])
        conn.commit()

    cursor.close()
    conn.close()

if __name__ == "__main__":
    generate_data(simulation_id=2)

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
