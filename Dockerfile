FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build-base


ARG JAX_PLUGIN=cuda

## Basic system setup

ENV user=devpod
SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND=noninteractive \
    TERM=linux

ENV TERM=xterm-color

ENV LANGUAGE=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LC_CTYPE=en_US.UTF-8 \
    LC_MESSAGES=en_US.UTF-8

RUN apt update && apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gpg \
        gpg-agent \
        less \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        llvm \
        locales \
        tk-dev \
        tzdata \
        unzip \
        vim \
        wget \
        xz-utils \
        zlib1g-dev \
        zstd \
    && sed -i "s/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g" /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && apt clean


## System packages

RUN apt-get update
RUN apt-get install -y git openssh-server
RUN apt-get install -y python3 python3-pip python-is-python3


## Add user & enable sudo

RUN useradd -ms /bin/bash ${user}
RUN usermod -aG sudo ${user}

RUN apt-get install -y sudo
RUN echo "${user} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER ${user}
WORKDIR /home/${user}


## Python packages

RUN pip install --upgrade pip
RUN pip install wheel


# add specific version of JAX directry to the container
RUN pip install jax["${JAX_PLUGIN}"]==0.5.0

COPY --chown=${user}:${user} ./pyproject.toml /home/${user}/
RUN pip install pip-tools
RUN python -m piptools compile --extra dev -o requirements.txt pyproject.toml
RUN pip install -r requirements.txt


## Dev tools (should not be in pyproject.toml)

RUN pip install ipython seaborn datasets


RUN echo 'export HOME=/home/'$user >> /home/$user/.bashrc
RUN echo 'export PATH=/home/'$user'/.local/bin:${PATH}' >> /home/$user/.bashrc


CMD ["echo", "Balance is the key"]