#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}  # modified: 29500
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

# e.g. CUDA_HOME=/usr/local/cuda-11.4 CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/_vpt/KAIST-RGB-ViTDet/vitdet_mask-rcnn_vit-b-mae.py 2 --work-dir ./work_dirs/vitdet_mask-rcnn_vit-b-kaist-512x640/rgb_full_8-4 --cfg-options find_unused_parameters=True
