#!/bin/bash

docker run --rm -ti --name=kalexu97-dck_baseline -p 1316:1316 --cpus 1 -v /home/kalexu97:/workspace -v /mnt/local:/workspace/mnt/local kalexu97_dc_baseline
