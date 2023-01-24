FROM ubuntu:20.04


ENV PATH="/root/miniconda3/bin:${PATH}"

ARG PATH="/root/miniconda3/bin:${PATH}"


RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/SapienzaNLP/spring.git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash \
    && . /root/.bashrc \
    && conda update conda -y \
    && conda install python=3.8.12 pip -y \
    && /root/miniconda3/bin/pip install -r /spring/requirements.txt

WORKDIR /home/grandi/project/server/