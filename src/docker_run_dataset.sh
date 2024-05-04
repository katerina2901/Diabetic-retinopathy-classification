#!/bin/bash

docker run --rm -ti --name=docker_dataset_kalexu97 -p 1316:1316 --cpus=1 -v /home/kalexu97:/workspace -v /mnt/local:/workspace/mnt/local docker_dataset

