FROM nvidia/cudagl:10.2-devel-ubuntu18.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync \
    curl \
    wget \
    htop \
    git \
    openssh-server \
    nano \
    cmake \
    unzip \
    cmake \
    zip \
    python-opencv \
    vim \
    ffmpeg \
    tmux \
    freeglut3-dev


# RUN mkdir -p /opt/cudnn && cd /opt/cudnn
# # https://developer.nvidia.com/rdp/cudnn-download
# COPY docker/cudnn-10.1-linux-x64-v7.6.5.32.tgz cudnn.tgz
# RUN tar xvf cudnn.tgz && \
#     cp -P cuda/include/cudnn.h /usr/local/cuda/include && \
#     cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 && \
#     chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=7.6.5.32-1+cuda10.1 \
    libcudnn7-dev=7.6.5.32-1+cuda10.1 \
    && apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# fixuid
ARG USERNAME=user
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER $USERNAME:$USERNAME

# miniconda
WORKDIR /home/user
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH

# RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh && \
    conda clean -ya

# python libs
RUN pip install --upgrade pip

## requirements
COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install -r requirements.txt

# valhitov smplx
COPY third_party/smplx smplx
RUN ls
RUN cd smplx && \
    pip --no-cache-dir install . && \
    cd ..

# docker setup
WORKDIR /
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]

