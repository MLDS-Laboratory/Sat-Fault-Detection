# Base image using Ubuntu with LTS support for Basilisk
FROM ubuntu:22.04

# Update current software
RUN apt-get update

# Install necessary packages for Basilisk and PostgreSQL
RUN apt-get install -y git build-essential python3 python3-setuptools python3-dev python3-pip python3.10-venv swig cmake \
    libpq-dev postgresql postgresql-contrib

# Clone the Basilisk repository
RUN git clone https://github.com/AVSLab/basilisk.git 
WORKDIR /basilisk

# Create a Python virtual environment for Basilisk
RUN python3 -m venv .venv

# Activate the virtual environment and install pip dependencies for Basilisk
RUN /bin/bash -c "source .venv/bin/activate && pip install --upgrade pip && pip install wheel 'conan<2.0' cmake"

# Initialize the default Conan profile for Basilisk
RUN /bin/bash -c "source .venv/bin/activate && conan profile new default --detect"

# Update Conan profile settings for Basilisk
RUN /bin/bash -c "source .venv/bin/activate && conan profile update settings.compiler.libcxx=libstdc++11 default"

# Install dependencies with Conan for Basilisk
RUN /bin/bash -c "source .venv/bin/activate && conan install . --build=missing -s build_type=Release"

# Run the Basilisk build
RUN /bin/bash -c "source .venv/bin/activate && /basilisk/.venv/bin/python3 conanfile.py"

# Install Python dependencies for PostgreSQL and additional requirements
RUN /bin/bash -c "source .venv/bin/activate && pip install psycopg2-binary"

# Setup PostgreSQL environment variables
ENV POSTGRES_HOST=localhost
ENV POSTGRES_DB=telemetry_db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Ensure the PostgreSQL service is started
RUN service postgresql start && \
    su - postgres -c "psql -c \"CREATE DATABASE telemetry_db;\"" && \
    su - postgres -c "psql -c \"CREATE USER postgres WITH PASSWORD 'postgres';\"" && \
    su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE telemetry_db TO postgres;\""

# Copy the data generator script to the container
COPY data_generator.py /app/data_generator.py

# Make sure the working directory is /basilisk
WORKDIR /basilisk

# Default command to run Bash and allow user interaction with both Basilisk and PostgreSQL
CMD ["/bin/bash"]
