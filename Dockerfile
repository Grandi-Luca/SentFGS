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

COPY ./amr-utils amr-utils

RUN git clone https://github.com/SapienzaNLP/spring.git

RUN mkdir documents

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash \
    && . /root/.bashrc \
    && conda update conda -y \
    && conda install pip python -y \
    && conda create -n spring-env python=3.8.12 pip -y \
    && /root/miniconda3/bin/pip install ctc_score summac \
    && /root/miniconda3/bin/pip install wandb penman spacy pandas torch transformers==4.24.0 \
    && python -m spacy download en_core_web_sm \
    && /root/miniconda3/bin/pip install smatch numpy scipy networkx gensim pyemd /amr-utils \
    && /root/miniconda3/envs/spring-env/bin/pip install -r /spring/requirements.txt