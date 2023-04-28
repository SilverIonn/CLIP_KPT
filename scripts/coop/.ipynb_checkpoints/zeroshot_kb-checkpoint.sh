#!/bin/bash

# custom config
DATA=/ix/yufeihuang/jia/nlp/prompt/data
TRAINER=ZeroshotCLIP3
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16



# for i in $(seq 0.1 0.1 1) 
# do

CUDA_VISIBLE_DEVICES=0 python train_withKB.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only \
--verbalizer \
--vb-dir /ihome/yufeihuang/jiy130/Prompt/CoOp/knowledgebase \
--vbsize 0 \
--calibration \
# --cutrate 0.15 \

# done