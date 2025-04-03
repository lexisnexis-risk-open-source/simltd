# Installation Guide
As a prerequisite, we first clone the SimLTD repository to our local home directory:

```bash
git clone \
    https://github.com/lexisnexis-risk-open-source/simltd.git \
    ~/simltd
# We create some new directories inside simltd
# to store development artifacts.
cd ~/simltd
mkdir results work_dirs
```

## Best Practices with Docker
We recommend using Docker to containerize all the complex dependencies when building this project. We provide a reference [Dockerfile](https://github.com/lexisnexis-risk-open-source/simltd/tree/main/docs/installation/docker/simltd/Dockerfile) as an example.

Make sure the GPU driver satisfies the minimum version requirements, according to [these NVIDIA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Also, ensure that the [docker version](https://docs.docker.com/engine/install/) is >=19.03.

### Build

```bash
# The reference build uses Python 3.10.13,
# PyTorch 2.1.2, CUDA 11.8, cuDNN 8, and MMDet 3.2.0.
docker build \
    --build-arg USER_ID=$(id -u) \
    -t simltd:pytorch2.1.2-cuda11.8-mmdet3.2.0 \
    -f docs/installation/docker/simltd/Dockerfile .
```

### Usage
We recommend running Docker as a user mapped from our local machine to the container via the argument `-u $(id -u)`, where the `bash` command `id -u` gives the user ID on the local host. Below is an example `docker run` command to execute a SimLTD training job on the LVIS dataset using 8 GPUs.

```bash
LOCAL_HOME_DIR=~
APP_HOME_DIR=/home/appuser
DATA_DIR=/data
docker run \
    -w ${APP_HOME_DIR}/simltd \
    --gpus='"device=0,1"' \
    -u $(id -u) --rm --ipc=host \
    -v ${LOCAL_HOME_DIR}/simltd:${APP_HOME_DIR}/simltd \
    -v ${DATA_DIR}:${APP_HOME_DIR}/simltd/data \
    simltd:pytorch2.1.2-cuda11.8-mmdet3.2.0 \
    bash tools/dist_train.sh \
    configs/simltd/dino-swin/dino-5scale_swin-t_lvis_v1_head866.py \
    8
```

Here, we assume that the LVIS data source (and others) is stored on the local machine at the path `/data/`. We use the `docker run -v` flag to map volumes between the local host and the container at runtime. Our recommended best practice is to map two volumes from the local host to be used by the container:

1. We map the entire local SimLTD repository to the container so that any local modifications will take effect inside the container and can be used by the container at runtime.

    ```bash
    -v ${LOCAL_HOME_DIR}/simltd:${APP_HOME_DIR}/simltd
    ```

2. We map *data volumes* to a specified location inside the container so it can access data not previously copied during runtime.

    ```bash
    -v ${DATA_DIR}:${APP_HOME_DIR}/simltd/data
    ```

Then, in our config files, we just need to point to the proper paths of data sources and other artifacts needed by the job, which are *relative to the working directory of the container*. The default working directory of the container is set by `docker run -w ${APP_HOME_DIR}/simltd`.

## Anaconda Environment
Alternatively, we can install SimLTD and its dependencies using an Anaconda environment.

**Step 1.** Download and install Anaconda from the [official website](https://www.anaconda.com/download).

**Step 2.** Create the conda environment.

```bash
# From inside the simltd working directory.
conda env create -f docs/installation/environment-cpu.yaml
```

or

```bash
# From inside the simltd working directory.
conda env create -f docs/installation/environment-gpu.yaml
```

**Step 3.** Verify the installation.

```bash
conda activate simltd-gpu

# No import error and with expected output: 3.2.0
python -c "import mmdet; print(mmdet.__version__)"
```
