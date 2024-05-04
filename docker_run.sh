#!/bin/bash

docker run --rm --name=test_kalexu97 -p 1311:1311 --gpus 1  -v ~/Projects/DeepLearning/Project/Docker_test/:/root/docker_test -ti docker_test # -c "cd src && source activate ml_test && python train.py"77
