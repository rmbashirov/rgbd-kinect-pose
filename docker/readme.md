Run 2 scripts to build final docker image, your nvidia driver should support cuda 10.2:
1) ~3 min, you will be prompted to accept eula manually, 
from `docker/accept_eula` dir run `./build.sh`
2) can take more than 1 hour, from `docker` dir run `./build.sh`
