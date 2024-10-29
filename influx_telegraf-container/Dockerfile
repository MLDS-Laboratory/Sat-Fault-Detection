# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y wget gnupg curl software-properties-common supervisor && \
    rm -rf /var/lib/apt/lists/*

# Add InfluxData repository GPG key and store it in /usr/share/keyrings
RUN wget -qO- https://repos.influxdata.com/influxdb.key | gpg --dearmor | tee /usr/share/keyrings/influxdb-archive-keyring.gpg > /dev/null

# Add InfluxData repository without signed-by (not ideal for prod, but works for dev)
RUN echo "deb [trusted=yes] https://repos.influxdata.com/ubuntu focal stable" | tee /etc/apt/sources.list.d/influxdb.list

# Update package lists with the new repository
RUN apt-get update

# Install InfluxDB and Telegraf
RUN apt-get install -y influxdb telegraf && \
    rm -rf /var/lib/apt/lists/*

# Copy InfluxDB and Telegraf configuration files
COPY influxdb.conf /etc/influxdb/influxdb.conf
COPY telegraf.conf /etc/telegraf/telegraf.conf

# Expose InfluxDB port
EXPOSE 8086

# Configure supervisord to run both InfluxDB and Telegraf
RUN echo "[supervisord]\nnodaemon=true\n\n[program:influxdb]\ncommand=/usr/bin/influxd\n\n[program:telegraf]\ncommand=/usr/bin/telegraf --config /etc/telegraf/telegraf.conf" > /etc/supervisor/conf.d/supervisord.conf

# Start supervisord to run both InfluxDB and Telegraf
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
