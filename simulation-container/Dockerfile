# LTS support
FROM ubuntu:22.04

# Update current software
RUN apt-get update

# Install necessary packages
RUN apt-get install -y git build-essential python3 python3-setuptools python3-dev python3-pip python3.10-venv swig cmake

# Clone the Basilisk repository
RUN git clone https://github.com/AVSLab/basilisk.git 
WORKDIR /basilisk

# Create a Python virtual environment
RUN python3 -m venv .venv

# Activate the virtual environment and install pip dependencies
RUN /bin/bash -c "source .venv/bin/activate && pip install --upgrade pip && pip install wheel 'conan<2.0' cmake"

# Initialize the default Conan profile
RUN /bin/bash -c "source .venv/bin/activate && conan profile new default --detect"

# Update Conan profile settings
RUN /bin/bash -c "source .venv/bin/activate && conan profile update settings.compiler.libcxx=libstdc++11 default"

# Install dependencies with Conan
RUN /bin/bash -c "source .venv/bin/activate && conan install . --build=missing -s build_type=Release"

# Run the Basilisk build
RUN /bin/bash -c "source .venv/bin/activate && /basilisk/.venv/bin/python3 conanfile.py"

# Default command
CMD ["/bin/bash"]
