#!/bin/bash

# 安装构建工具
# Install build tools
# conda install -c conda-forge gcc=14
# conda install -c conda-forge gxx
conda install ffmpeg  # cmake

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# 刷新环境
# Refresh environment
hash -r

/root/miniconda3/envs/GPTSoVits/bin/pip install -r requirements.txt # 显式指定pip路径，不然我的WSL2会报错：External Environments Management