#!/bin/bash

docker build . \
    -f Dockerfile \
    --build-arg USER=${USER} \
    --build-arg UID=${UID} \
    --build-arg GID=${UID} \
    -t fiqa:latest
