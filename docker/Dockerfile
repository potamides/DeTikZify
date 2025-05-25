FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        git \
        wget \
        texlive-full \
        ghostscript \
        poppler-utils \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py


WORKDIR /app
RUN <<EOT 
#!/bin/bash
pip install --upgrade pip setuptools
pip install --no-cache-dir hf_xet
git clone https://github.com/potamides/DeTikZify /tmp/DeTikZify
cd /tmp/DeTikZify
cat <<EOF > setup.py
from setuptools import setup

setup()
EOF
pip install --no-cache-dir .[examples,legacy]
rm -rf /tmp/DeTikZify
EOT

ENV PYTHONPATH=/app
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

ENTRYPOINT ["python3"]
