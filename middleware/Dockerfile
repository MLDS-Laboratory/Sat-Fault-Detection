FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir psycopg2-binary kafka-python

COPY middleware.py ./

ENV POSTGRES_HOST=postgres
ENV POSTGRES_DB=telemetry_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092

CMD ["python", "middleware.py"]
