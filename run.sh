#! /bin/bash

nohup python -u main.py --gpuId 0 --testSize 0.02 --num4Epoches 600 1>./files/info0.txt 2>&1 &
nohup python -u main.py --gpuId 1 --testSize 0.1 --num4Epoches 1000 1>./files/info1.txt 2>&1 &
