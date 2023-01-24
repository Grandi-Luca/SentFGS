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

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash \
    && . /root/.bashrc \
    && conda update conda -y \
    && conda install python=3.8.12 -y \
    && /root/miniconda3/bin/pip install summac ctc_score \
    && /root/miniconda3/bin/pip install wandb numpy scipy pandas \
    && /root/miniconda3/bin/pip install penman spacy torch transformers==4.24.0  \
    && python -m spacy download en_core_web_sm \
    && /root/miniconda3/bin/pip install networkx gensim pyemd /amr-utils smatch

WORKDIR /home/grandi/project/client/