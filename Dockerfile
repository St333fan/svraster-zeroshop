FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL description="Docker container for SVRaster with CUDA 12.4 support"
ARG DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -ay

# Initialize conda
RUN conda init bash

RUN mkdir -p /home/appuser/svraster
COPY . /home/appuser/svraster/

WORKDIR /home/appuser/svraster

# Create main svraster environment
RUN conda create -n svraster python=3.9 -y && \
    echo "source activate svraster" > ~/.bashrc

# Install PyTorch and dependencies in svraster environment
RUN /bin/bash -c "source activate svraster && \
    conda install pytorch==2.5.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e cuda/"

# Create postprocess environment
RUN conda create -n postprocess python=3.10 -y && \
    /bin/bash -c "source activate postprocess && \
    conda install -c conda-forge pymeshlab -y && \
    pip install --no-cache-dir scipy"

# Set default environment to svraster
ENV CONDA_DEFAULT_ENV=svraster

# Activate svraster environment by default
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash", "-c", "source activate svraster && exec bash"]
