# no cudnn here
FROM nvidia/cudagl:10.2-devel-ubuntu18.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y curl && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/18.04/prod bionic main" >> /etc/apt/sources.list && \
    apt-get update && \
    env DEBIAN_FRONTEND=noninteractive && \
    echo 'libk4a1.3 libk4a1.3/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | debconf-set-selections && \
    echo 'libk4a1.3 libk4a1.3/accept-eula boolean true' | debconf-set-selections && \
    apt-get install -y libk4a1.3 libk4a1.3-dev

    # # this does not work, so we have to accept eula manually
    # echo 'libk4abt1.0 libk4abt1.0/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | debconf-set-selections && \
    # echo 'libk4abt1.0 libk4abt1.0/accept-eula boolean true' | debconf-set-selections && \
    # apt-get install -y libk4abt1.0 libk4abt1.0-dev

COPY install_k4abt.sh /opt/install_k4abt.sh
WORKDIR /opt
