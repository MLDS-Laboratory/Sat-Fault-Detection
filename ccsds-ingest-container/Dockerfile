FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install confluent-kafka influxdb
CMD ["python", "ccsds_ingest.py"]
