# LTS support
FROM ubuntu:20.04

# non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# required packages
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    swig \
    cmake \
    python3-tk \
    && apt-get clean

# clone the Basilisk repository
RUN git clone https://github.com/AVSLab/basilisk.git /opt/basilisk
WORKDIR /opt/basilisk

# python venv setup
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip

# use conan to install https://hanspeterschaub.info/basilisk/Install/installOnLinux.html
RUN . .venv/bin/activate && \
    pip install wheel 'conan<2.0' cmake \
    pandas matplotlib numpy<=2.0.1 colorama tqdm Pillow pytest pytest-html pytest-xdist

# Conan to build Basilisk
RUN . .venv/bin/activate && \
    python3 conanfile.py

# Entry point to ensure the virtual environment is activated when running commands
ENTRYPOINT ["/bin/bash", "-c", "source .venv/bin/activate && exec \"$@\"", "--"]
