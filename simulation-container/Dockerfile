# Base image using Ubuntu with LTS support for Basilisk
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Update current software
RUN apt-get update

# Set timezone and locale environment variables
ENV TZ America/New_York 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Preconfigure locales
RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install necessary packages for Basilisk / PostgreSQL / x11
RUN apt-get update && \
    apt-get install -y git build-essential python3 python3-setuptools python3-dev python3-pip python3.10-venv swig cmake \
    libpq-dev postgresql postgresql-contrib x11-apps python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Clone the Basilisk repository
RUN git clone https://github.com/AVSLab/basilisk.git 
WORKDIR /basilisk

# Create a Python virtual environment for Basilisk
RUN python3 -m venv .venv

# Activate the virtual environment and install pip dependencies for Basilisk
RUN /bin/bash -c "source .venv/bin/activate && pip install --upgrade pip && pip install wheel 'conan' cmake packaging"

# Initialize the default Conan profile for Basilisk
RUN /bin/bash -c "source .venv/bin/activate && conan profile detect"

# Update Conan profile settings for Basilisk
RUN /bin/bash -c "source .venv/bin/activate && echo 'compiler.libcxx=libstdc++11' >> ~/.conan2/profiles/default"

RUN /bin/bash -c "source .venv/bin/activate && conan install . --build=missing -s build_type=Release"

# Run the Basilisk build
RUN /bin/bash -c "source .venv/bin/activate && /basilisk/.venv/bin/python3 conanfile.py"

# Install Python dependencies for PostgreSQL and additional requirements
RUN /bin/bash -c "source .venv/bin/activate && pip install psycopg2-binary kafka-python"

RUN apt-get update && \
    apt-get install -y supervisor software-properties-common curl gnupg

# Copy supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Modify pg_hba.conf to use trust authentication for all connections
RUN sed -i "s/peer/trust/g" /etc/postgresql/14/main/pg_hba.conf && \
    sed -i "s/scram-sha-256/trust/g" /etc/postgresql/14/main/pg_hba.conf

# Start PostgreSQL service and set up the database and user
RUN service postgresql start && \
    su - postgres -c "psql -c \"CREATE DATABASE telemetry_db;\"" && \
    su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='postgres'\" | grep -q 1 || psql -c \"CREATE USER postgres WITH PASSWORD 'postgres';\"" && \
    su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE telemetry_db TO postgres;\""


# Set the working directory
WORKDIR /basilisk

# Copy middleware script into the container
COPY middleware.py /basilisk/middleware.py

# Expose PostgreSQL port
EXPOSE 5432

# Automatically activate the virtual environment in any new shell
RUN echo "source /basilisk/.venv/bin/activate" >> /root/.bashrc

# Start supervisord to manage PostgreSQL
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
