# no cudnn here
# warning: base image name hardcoded, check it matches NAME_02 from source.sh
FROM azure_kinect:02_eula

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
    freeglut3-dev


RUN apt-get update && apt-get install -y \
    pkg-config \
    ninja-build \
    doxygen \
    clang \
    gcc-multilib \
    g++-multilib \
    python3 \
    git-lfs \
    nasm \
    cmake \
    libgl1-mesa-dev \
    libsoundio-dev \
    libvulkan-dev \
    libx11-dev \
    libxcursor-dev \
    libxinerama-dev \
    libxrandr-dev \
    libusb-1.0-0-dev \
    libssl-dev \
    libudev-dev \
    mesa-common-dev \
    uuid-dev


# cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=7.6.5.32-1+cuda10.2 \
    libcudnn7-dev=7.6.5.32-1+cuda10.2 \
    && apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


# make tf==1.15 built for cuda 10.0 work with cuda 10.2
# remove cudnn 7.3, so that only cudnn 7.6.5 exists
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/libcufft.so.10 /usr/local/cuda/lib64/libcufft.so.10.0 && \
    ln -s /usr/local/cuda/lib64/libcurand.so.10 /usr/local/cuda/lib64/libcurand.so.10.0 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so.10 /usr/local/cuda/lib64/libcusolver.so.10.0 && \
    ln -s /usr/local/cuda/lib64/libcusparse.so.10 /usr/local/cuda/lib64/libcusparse.so.10.0 && \
    rm -rf /usr/lib/libcudnn.so.7


# fixuid to allow user to set container's user/group id
# don't forget to set ENTRYPOINT and/or CMD
ARG USERNAME=docker
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


# conda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# this version of miniconda's /opt/conda/bin provides pip = pip3 = pip3.7, python = python3 = python3.7
ENV PATH /opt/conda/bin:$PATH
RUN pip install --upgrade pip && pip install numpy==1.19.2


# azure sdk
RUN git clone --recursive https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git /opt/azure-sdk && \
    cd /opt/azure-sdk && \
    git checkout 17b644560ce7b4ee7dd921dfff0ae811aa54ede6 && \
    mkdir -p /etc/udev/rules.d && \
    cp ./scripts/99-k4a.rules /etc/udev/rules.d/ && \
    mkdir build && cd build && \
    cmake .. -GNinja && \
    ninja


# opencv
RUN apt update && \
    apt install -y \
        build-essential cmake pkg-config unzip yasm git checkinstall \
        libturbojpeg0-dev \
        libavcodec-dev libavformat-dev libswscale-dev libavresample-dev \
        libjpeg-dev libpng-dev libtiff-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
        libfaac-dev libmp3lame-dev libvorbis-dev \
        libgtk-3-dev \
        libtbb-dev \
        libatlas-base-dev gfortran \
        libprotobuf-dev protobuf-compiler \
        libgoogle-glog-dev libgflags-dev \
        libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
RUN cd /opt && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip && \
    wget -O opencv_contrib.zip https://github.com//opencv/opencv_contrib/archive/4.2.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip
RUN cd /opt/opencv-4.2.0 && \
    mkdir -p build && cd build && \
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D WITH_TBB=ON \
        -D WITH_CUDA=OFF \
        -D BUILD_opencv_cudacodec=OFF \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=0 \
        -D WITH_V4L=OFF \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=OFF \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_PC_FILE_NAME=opencv.pc \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D BUILD_opencv_java=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON_LIBRARY=$(python -c "from distutils.sysconfig import get_python_lib, get_python_version; import os.path as osp; lib_dp=osp.abspath(osp.join(get_python_lib(), '..', '..')); lib_fp=osp.join(lib_dp, f'libpython{get_python_version()}m.so'); print(lib_fp);") \
        -D PYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        -D OPENCV_PYTHON_INSTALL_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        -D PYTHON_EXECUTABLE=$(which python) \
        -D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.2.0/modules" \
        -D BUILD_EXAMPLES=OFF \
        ..
RUN cd /opt/opencv-4.2.0/build && \
    make -j$(nproc) && \
    make install


# python pkgs
# pip installs torch with cuda 10.2 by default, cannot specify cuda version
# use conda to install torch
RUN conda install -y pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=10.2 -c pytorch
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt



############## Install from extern folder ##############

# minimal_pytorch_rasterizer for fast mesh visualization, requires nvcc
ENV TORCH_CUDA_ARCH_LIST="Maxwell Pascal Volta Turing"
COPY extern/minimal_pytorch_rasterizer /opt/minimal_pytorch_rasterizer
RUN cd /opt/minimal_pytorch_rasterizer && ./setup.sh

# nlohmann_json for pyk4a
RUN cd /opt && \
    git clone --recursive https://github.com/nlohmann/json ./nlohmann_json && \
    cd nlohmann_json && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build . -- -j$(nproc) && \
    make install
# pyk4a: python wrapper for azure kinect streaming
COPY extern/pyk4a /opt/extern/pyk4a
RUN cd /opt/extern/pyk4a && ./setup.sh
RUN apt install -y usbutils
# prevent 100% cpu utilization bug when using kinect body tracking
ENV OMP_WAIT_POLICY Passive

# multiprocessing_pipeline to run blocks in parallel processes
COPY extern/multiprocessing_pipeline /opt/multiprocessing_pipeline
RUN cd /opt/multiprocessing_pipeline && ./setup.sh

# patched_smplx - our smplx with kinect joints
COPY extern/patched_smplx /opt/extern/patched_smplx
RUN cd /opt/extern/patched_smplx && pip --no-cache-dir install .

# smplx_optimization - dependency of smplx_kinect
COPY extern/smplx_optimization /opt/smplx_optimization
RUN cd /opt/smplx_optimization && ./setup.sh

# smplx_kinect to load body model
COPY extern/smplx_kinect /opt/smplx_kinect
RUN cd /opt/smplx_kinect && ./setup.sh

# face_expression to load face model
RUN pip install --no-cache-dir --no-deps kornia==0.3.0
COPY extern/face_expression /opt/face_expression
RUN cd /opt/face_expression && pip --no-cache-dir install .

# minimal_hand to load hand model
COPY extern/minimal_hand /opt/minimal_hand
RUN cd /opt/minimal_hand && pip --no-cache-dir install .

############## Install from extern folder ##############


USER $USERNAME:$USERNAME
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]
WORKDIR /src
