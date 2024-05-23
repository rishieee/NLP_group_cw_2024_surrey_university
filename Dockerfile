# Use a specific NVIDIA CUDA and Ubuntu version as a base image
ARG BASE_IMAGE=nvidia/cuda:12.4.1-base-ubuntu22.04
ARG PYTHON_VERSION=3.11
ARG MODEL="quantized"

FROM ${BASE_IMAGE} as dev_base

ENV DEBIAN_FRONTEND=noninteractive

# Update the package lists and install essential packages
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        unzip

# Set the working directory to root (adjusted to be more explicit)
WORKDIR /root

# Copy the requirements file into the image
COPY requirements.txt requirements.txt

# Install Python packages from the requirements file
RUN python3 -m pip install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch, torchvision, and torchaudio for the specific CUDA version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the English language model for spaCy
RUN python3 -m spacy download en_core_web_sm

# Install NLTK and download required NLTK data
RUN pip3 install nltk
RUN python3 -c "import nltk; nltk.download('punkt')"

# Install Gunicorn
RUN pip3 install gunicorn

RUN mkdir web_app
WORKDIR /root/web_app/

COPY web_app/model_predict.py model_predict.py
COPY web_app/nlp_group.py nlp_group.py 
COPY web_app/process.py process.py
COPY web_app/gunicorn_config.py gunicorn_config.py

COPY models.zip models.zip

# Unzip and organize model files
RUN unzip models.zip -d models && \
    mv models/models/* models && \
    rm -rf models/models models.zip

ADD . db

# Set the MODEL environment variable
ENV MODEL=${MODEL}

# The command to run the web application script
#ENTRYPOINT ["python3", "-W", "ignore", "nlp_group.py", "${MODEL}"]
ENTRYPOINT ["gunicorn", "-c", "gunicorn_config.py", "nlp_group:app"]



