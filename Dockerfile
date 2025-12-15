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
    libgl1-mesa-dri \
    mesa-utils \
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

# Accept conda Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN mkdir -p /home/appuser/svraster
COPY . /home/appuser/svraster/

WORKDIR /home/appuser/svraster

# Create main svraster environment
RUN conda create -n svraster python=3.9 -y && \
    echo "source activate svraster" > ~/.bashrc

# Install PyTorch with CUDA support FIRST (before other requirements)
# Use pip with explicit CUDA wheel to avoid XPU detection issues
RUN /bin/bash -c "source activate svraster && \
    pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 --index-url https://download.pytorch.org/whl/cu124"

# Install ninja for faster builds
RUN /bin/bash -c "source activate svraster && \
    pip install ninja"

# Install fused-ssim: clone and patch setup.py to force CUDA
RUN /bin/bash -c "source activate svraster && \
    git clone https://github.com/rahul-goel/fused-ssim.git /tmp/fused-ssim && \
    cd /tmp/fused-ssim && \
    sed -i 's/if torch.cuda.is_available():/if True:  # Force CUDA/' setup.py && \
    CUDA_ARCHITECTURES='75;80;86;89' pip install --no-build-isolation . && \
    rm -rf /tmp/fused-ssim"

# Install other dependencies (excluding fused-ssim which is already installed)
RUN /bin/bash -c "source activate svraster && \
    pip install --no-cache-dir -r requirements.txt"

# Install CUDA extensions (--no-build-isolation to use existing torch)
# Set TORCH_CUDA_ARCH_LIST to avoid GPU detection during build
RUN /bin/bash -c "source activate svraster && \
    TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;9.0' pip install --no-build-isolation --no-cache-dir -e cuda/"

# Create postprocess environment
RUN conda create -n postprocess python=3.10 -y && \
    /bin/bash -c "source activate postprocess && \
    conda install -c conda-forge pymeshlab -y && \
    pip install --no-cache-dir scipy"

# Set default environment to svraster
ENV CONDA_DEFAULT_ENV=svraster

# Rebuild CUDA extension to ensure _C.so is compiled
RUN /bin/bash -c "source activate svraster && \
    cd /home/appuser/svraster/cuda && \
    pip uninstall -y svraster_cuda && \
    TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;9.0' pip install --no-build-isolation --no-cache-dir -e ."

# Activate svraster environment by default
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash", "-c", "source activate svraster && exec bash"]
