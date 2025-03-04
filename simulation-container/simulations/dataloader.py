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

    times, pos, velo, sigma, omega, CSSdata, motorTorque, thrLog, sensedSun, sunPoint = sc.run(False, False, True, False)
    times2, pos2, velo2, sigma2, omega2, CSSdata2, motorTorque2, thrLog2, sensedSun2, sunPoint2 = sc.run(False, False, True, False)
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

    for i in range(len(times)):

        data_payload = {
            "x_pos": pos[i, 0],
            "y_pos": pos[i, 1],
            "z_pos": pos[i, 2],
            "x_velo": velo[i, 0],
            "y_velo": velo[i, 1],
            "z_velo": velo[i, 2],
            "x_sigma": sigma[i, 0],
            "y_sigma": sigma[i, 1],
            "z_sigma": sigma[i, 2],
            "x_omega": omega[i, 0],
            "y_omega": omega[i, 1],
            "z_omega": omega[i, 2],
            "CSS_1": CSSdata[0, i],
            "CSS_2": CSSdata[1, i],
            "CSS_3": CSSdata[2, i],
            "CSS_4": CSSdata[3, i],
            "CSS_5": CSSdata[4, i],
            "CSS_6": CSSdata[5, i],
            "RW_1": motorTorque[0][i],
            "RW_2": motorTorque[1][i],
            "RW_3": motorTorque[2][i],
            "Thruster_1": thrLog[0][i],
            "Thruster_2": thrLog[1][i],
            "Thruster_3": thrLog[2][i],
            "sun_x_sensed": sensedSun[i][0],
            "sun_y_sensed": sensedSun[i][1],
            "sun_z_sensed": sensedSun[i][2],
            "sun_x_true": sunPoint[i, 0],
            "sun_y_true": sunPoint[i, 1],
            "sun_z_true": sunPoint[i, 2]
        }

        data_payload2 = {
            "x_pos": pos2[i, 0],
            "y_pos": pos2[i, 1],
            "z_pos": pos2[i, 2],
            "x_velo": velo2[i, 0],
            "y_velo": velo2[i, 1],
            "z_velo": velo2[i, 2],
            "x_sigma": sigma2[i, 0],
            "y_sigma": sigma2[i, 1],
            "z_sigma": sigma2[i, 2],
            "x_omega": omega2[i, 0],
            "y_omega": omega2[i, 1],
            "z_omega": omega2[i, 2],
            "CSS_1": CSSdata2[0, i],
            "CSS_2": CSSdata2[1, i],
            "CSS_3": CSSdata2[2, i],
            "CSS_4": CSSdata2[3, i],
            "CSS_5": CSSdata2[4, i],
            "CSS_6": CSSdata2[5, i],
            "RW_1": motorTorque2[0][i],
            "RW_2": motorTorque2[1][i],
            "RW_3": motorTorque2[2][i],
            "Thruster_1": thrLog2[0][i],
            "Thruster_2": thrLog2[1][i],
            "Thruster_3": thrLog2[2][i],
            "sun_x_sensed": sensedSun2[i][0],
            "sun_y_sensed": sensedSun2[i][1],
            "sun_z_sensed": sensedSun2[i][2],
            "sun_x_true": sunPoint2[i, 0],
            "sun_y_true": sunPoint2[i, 1],
            "sun_z_true": sunPoint2[i, 2]
        }

        cursor.execute(sql.SQL("""
                INSERT INTO {} (time, satellite_id, data)
                VALUES (%s, %s, %s);
            """).format(sql.Identifier(table_name)), (i*5, 1, json.dumps(data_payload)))
        conn.commit()

        cursor.execute(sql.SQL("""
                INSERT INTO {} (time, satellite_id, data)
                VALUES (%s, %s, %s);
            """).format(sql.Identifier(table_name)), (i*5, 2, json.dumps(data_payload2)))

    cursor.close()
    conn.close()

if __name__ == "__main__":
    generate_data(simulation_id=3042025)

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
