#! /bin/bash

nohup python -u main.py --gpuId 0 --num4Epoches 100 1>./files/info0.txt 2>&1 &
nohup python -u main.py --gpuId 1 --num4Epoches 150 1>./files/info1.txt 2>&1 &
#nohup python -u main.py --gpuId 2 --num4Epoches 200 1>./files/info2.txt 2>&1 &
nohup python -u main.py --gpuId 2 --num4Epoches 200 1>./files/info2.txt 2>&1 &
