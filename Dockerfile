FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y htop python3-dev wget

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n ml_test python=3.6

COPY . src/
RUN /bin/bash -c "cd src \
    && source activate ml_test \
    && pip install -r requirements.txt"
 