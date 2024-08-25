# It's a bit cumbersome, but we follow a three step approach for building the singularity image (locally)
# - Build image using podman: podman build --tag [IMAGE:TAG] -f Dockerfile .
# - Convert to OCI tar archive: podman save --format=oci-archive [IMAGE:TAG] -o [OUTPUT.tar]
# - Convert OCI archive to singularity image: singularity build [OUTPUT.sif] oci-archive://[OUTPUT.tar]

# Basic cuda image, works on most GPU architectures we're currently using
# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Something with jax, but not the right one
#FROM nvcr.io/nvidia/jax:23.08-py3

# See https://github.com/NVIDIA/JAX-Toolbox, we might need ghcr.io/nvidia/jax:gemma to run on A/V100, however
FROM ghcr.io/nvidia/jax:jax

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \ 
        python3-pip \
        curl \
        unzip \ 
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --upgrade pip \ 
    && pip3 install --upgrade -r requirements.txt


