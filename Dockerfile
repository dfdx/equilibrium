# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build-base
FROM ubuntu:24.04 AS build-base


## Basic system setup

SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND=noninteractive \
    TERM=linux

ENV TERM=xterm-color

ENV LANGUAGE=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LC_CTYPE=en_US.UTF-8 \
    LC_MESSAGES=en_US.UTF-8

RUN apt-get update && apt install -y --no-install-recommends \
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

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    python-is-python3 \
    python3 \
    python3-pip \
    && apt-get clean \
    && pip install uv --break-system-packages

## Add user & enable sudo

ENV user=devpod

RUN useradd -ms /bin/bash ${user} \
    && usermod -aG sudo ${user} \
    && apt-get install -y sudo \
    && apt-get clean \
    && echo "${user} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && echo 'export PATH=${PATH}:~/.local/bin' >> /home/${user}/.bashrc

USER ${user}
WORKDIR /home/${user}

## Python packages

# avoid uv warning related to Windows/Linux compatibility issues
ENV UV_LINK_MODE=copy

# create globally visible venv
# also set $VIRTUAL_ENV which will be used by uv
ENV VIRTUAL_ENV=/venv
RUN sudo mkdir "$VIRTUAL_ENV" \
    && sudo chown -R ${user}:${user} "$VIRTUAL_ENV"

# install the project
ENV BUILD_DIR=/app
COPY --chown=${user}:${user} . "$BUILD_DIR"


WORKDIR "${BUILD_DIR}"
RUN uv lock && uv sync --active
WORKDIR /home/${user}

# Install specific variation of JAX, but don't add to prooject dependencies
RUN uv pip install jax[cuda]==0.6.0


###########################################################
FROM build-base AS build-dev

## Other tools

# create a shortcut for uv run --active
RUN echo 'alias uvrun="uv run --active"' >> /home/${user}/.bashrc


CMD ["echo", "Keep the balance!"]
