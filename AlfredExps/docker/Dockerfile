# Dependicies are based on 
# https://github.com/askforalfred/alfred/blob/master/scripts/install_deps.sh

FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    # General
    apt-utils \
    sudo \
    p7zip-full \
    tmux \
    openssh-server \
    # tree \
    less \
    vim \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    git \
    dialog \
    ffmpeg \
    nvidia-settings \
    libffi-dev \
    flex \
    bison \
    build-essential \
    # module-init-tools \ <-- Package 'module-init-tools' has no installation candidate
    pciutils \
    # Graphics
    libsm6 \
    libxext6 \
    libxrender-dev \
    mesa-utils \
    # nvidia-xconfig \ <-- nvidia-xconfig is not actually used
    xserver-xorg \
    xserver-xorg-video-fbdev \
    xauth \
    # Python
    python3-pip \
    python3.8-venv \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 22

# Add a default user
ARG USER=docker_user
ARG UID=1000
ARG GID=1000
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}
RUN chown -R ${UID}:${GID} /home/${USER}
USER ${UID}:${GID}
