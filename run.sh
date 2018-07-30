#! /bin/bash

nohup python -u main.py --gpuId 0 1>./files/info0.txt 2>&1 &
nohup python -u main.py --gpuId 1 1>./files/info1.txt 2>&1 &
