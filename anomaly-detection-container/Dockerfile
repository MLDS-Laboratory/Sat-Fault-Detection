FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the anomaly detection script and models
COPY anomaly_detection.py ./
COPY models ./models

# Environment variables
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV INFLUXDB_HOST=influx_telegraf
ENV INFLUXDB_PORT=8086
ENV INFLUXDB_DB=telemetry_db

# Command to run the anomaly detection script
CMD ["python", "anomaly_detection.py"]
