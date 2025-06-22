#!/bin/bash

current_dir=$(pwd)

docker run -itd --rm \
    --gpus all \
    --ipc host \
    --privileged \
    --add-host host.docker.internal:host-gateway \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${current_dir}/../:/home/${USER} \
    -p ${UID}0:22 \
    --name ${USER}.fiqa \
    fiqa:latest

docker exec --user root ${USER}.fiqa bash -c "/etc/init.d/ssh start"
