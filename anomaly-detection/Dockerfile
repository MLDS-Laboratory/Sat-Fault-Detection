FROM python:3.9-slim

WORKDIR /app


# no cache apparently saves space
RUN pip install --no-cache-dir kafka-python     

COPY anomaly_detection.py ./

ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092

CMD ["python", "anomaly_detection.py"]
