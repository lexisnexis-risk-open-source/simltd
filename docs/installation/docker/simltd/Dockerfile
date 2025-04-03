ARG PYTORCH="2.1.2"
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0+PTX;7.5;8.0;8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"
# The default location for Torch Hub models is ~/.cache/torch/
# We can specify a different location by uncommenting the line below.
ENV TORCH_HOME="data/pytorch_models/"
ENV DEBIAN_FRONTEND=noninteractive

# Update and install OS packages
RUN apt-get update && apt-get install -y ffmpeg wget vim git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root appuser
# Obtain your USER_ID from bash command `id -u`
ARG USER_ID
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"
WORKDIR /home/appuser

# Install mmlab dependencies and pin setuptools to avoid import error
RUN pip install --no-cache-dir --user --upgrade pip wheel
RUN pip install --no-cache-dir --user setuptools==69.5.1
RUN pip install --no-cache-dir --user openmim
RUN mim install "mmengine>=0.7.1,<1.0.0"
RUN mim install "mmcv>=2.0.0rc4,<2.2.0"

# Install dependencies and useful packages
RUN conda clean -y --all
RUN pip install --no-cache-dir --user git+https://github.com/lvis-dataset/lvis-api.git
RUN pip install --no-cache-dir --user imagesize seaborn jupyter tensorboard scikit-video scikit-learn pandas natsort

# Install SimLTD with MMDet v3.2.0
RUN git clone https://github.com/lexisnexis-risk-open-source/simltd.git
WORKDIR /home/appuser/simltd
RUN pip install --no-cache-dir --user -r requirements/build.txt
RUN pip install --no-cache-dir --user -r requirements/optional.txt
RUN pip install --no-cache-dir --user -r requirements/multimodal.txt
RUN pip install --no-cache-dir --user -r requirements/albu.txt
RUN pip install --no-cache-dir --user -r requirements/tracking.txt
RUN pip install --no-cache-dir --user -e .

