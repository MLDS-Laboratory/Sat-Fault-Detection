FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Install InfluxDB
RUN wget -qO- https://repos.influxdata.com/influxdb.key | apt-key add - && \
    echo "deb https://repos.influxdata.com/ubuntu focal stable" | tee /etc/apt/sources.list.d/influxdb.list && \
    apt-get update && \
    apt-get install -y influxdb && \
    rm -rf /var/lib/apt/lists/*

# Install Telegraf
RUN wget -qO- https://repos.influxdata.com/influxdb.key | apt-key add - && \
    echo "deb https://repos.influxdata.com/ubuntu focal stable" | tee /etc/apt/sources.list.d/influxdb.list && \
    apt-get update && \
    apt-get install -y telegraf && \
    rm -rf /var/lib/apt/lists/*

# Copy InfluxDB and Telegraf configurations
COPY influxdb.conf /etc/influxdb/influxdb.conf
COPY telegraf.conf /etc/telegraf/telegraf.conf

# Expose InfluxDB port
EXPOSE 8086

# Start InfluxDB and Telegraf
CMD service influxdb start && telegraf
