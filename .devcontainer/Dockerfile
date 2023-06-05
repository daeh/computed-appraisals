# Dockerfile

# FROM arm64v8/node:18-bullseye
FROM node:18-bullseye

# Docker default of `/bin/sh` doesn't support `source`
SHELL ["/bin/bash", "-c"]

### Install shell dependencies

RUN apt-get update
RUN apt-get install --no-install-recommends --yes wget bzip2 ca-certificates git tini

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir /imgbase
WORKDIR /imgbase

### Install TeXLive

# RUN apt-get update \
#     && apt-get install -y \
#     texlive-full \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

### :latest with apt-get
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends texlive-full
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
# RUN tlmgr paper letter

# Copy & setup TeXLive binary from installer
ENV PATH=/usr/local/bin/texlive:$PATH

### Install conda

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=22.9.0-1
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /imgbase/miniforge.sh
# wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

RUN ["chmod", "+x", "/imgbase/miniforge.sh"]
RUN /imgbase/miniforge.sh -b -p ${CONDA_DIR}

RUN rm /imgbase/miniforge.sh
RUN conda clean --tarballs --index-cache --packages --yes
RUN find ${CONDA_DIR} -follow -type f -name '*.a' -delete
RUN find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete
RUN conda clean --force-pkgs-dirs --all --yes

### Setup conda env

### Create conda env from lockfile
COPY conda-linux-64.lock /imgbase/conda-linux-64.lock
RUN conda create --name envcam --file /imgbase/conda-linux-64.lock
RUN conda clean --all --yes

RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate envcam" >> /etc/skel/.bashrc
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate envcam" >> ~/.bashrc

### Install WebPPL
RUN npm install -g webppl
RUN mkdir -p ~/.webppl
RUN npm install --prefix ~/.webppl webppl-json

###
### To run the model, use the following command in the github codespace terminal:
###
### python code/cam_main.py --projectdir $(pwd)
###

ENTRYPOINT ["/opt/conda/envs/envcam/bin/python"]