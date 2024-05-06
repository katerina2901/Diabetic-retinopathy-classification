FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
# FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y htop python3-dev wget tmux

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n virt_env python=3.6

COPY . src/
RUN /bin/bash -c "cd src \
    && source activate virt_env \
    && pip install -r requirements_kaggle.txt \
    && jupyter notebook --generate-config \
    && cp /src/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py"
 