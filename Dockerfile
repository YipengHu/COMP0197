# Dockerfile for the module tutorial and coursework

FROM ubuntu:20.04

# git and conda
RUN apt-get update && apt-get install -y wget git \
 && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && mkdir /root/.conda \
 && bash Miniconda3-latest-Linux-x86_64.sh -b \
 && rm -f Miniconda3-latest-Linux-x86_64.sh
ARG PATH="/root/miniconda3/bin:$PATH"
RUN conda init bash

# clone the repo in "/workspace"
RUN git clone https://github.com/yipenghu/COMP0090.git workspace/comp0090

# create the tutorial/coursework conda environment "comp0090"
RUN conda create -n comp0090 python=3.9 tensorflow=2.4 pytorch=1.7 torchvision=0.8

# start "comp0090" by default
ENV PATH="/root/miniconda3/bin:$PATH"
