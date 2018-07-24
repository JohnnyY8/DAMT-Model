#! /bin/bash

nohup python -u main.py --gpuId 0 --num4Epoches 100 1>./files/info0.txt 2>&1 &
nohup python -u main.py --gpuId 1 --num4Epoches 60 1>./files/info1.txt 2>&1 &
