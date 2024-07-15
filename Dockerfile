FROM nvcr.io/nvidia/pytorch:24.03-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    sudo \
    nano && \
    apt-get clean &&\
    rm -rf /var/lib/apt/lists/*

# RUN pip install opencv-contrib-python-headless --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools

RUN useradd -m ash
# next line might be redundant
RUN usermod -aG sudo ash
RUN chown -R ash:ash /home/ash/
USER ash

WORKDIR /home/ash

COPY --chown=ash datasets ./datasets
COPY --chown=ash improved_diff ./improved_diff
COPY --chown=ash loggs ./loggs
COPY --chown=ash src/image_train.py ./src/image_train.py
COPY --chown=ash setup.py .
COPY --chown=ash requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

