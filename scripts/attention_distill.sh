#!/bin/bash
# Attention Distillation: the student learns its own attention with a cross-entropy loss
# (weight --atd_weight) towards the teacher's attention maps of all layers.
# Model names are the same as in attention_copy.sh. Batch size 128 x 8 GPUs = 1024.
IMAGENET_PATH=${IMAGENET_PATH:-/path/to/imagenet}
TEACHER=${TEACHER:-/path/to/teacher_checkpoint.pth}
MODEL=${MODEL:-dual_vit_small_patch16}
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/distill_${MODEL}}

torchrun --nproc_per_node=8 main_finetune.py \
    --model ${MODEL} --mode distill --end_layer 0 --atd_weight 3.0 \
    --finetune ${TEACHER} \
    --epochs 20 --warmup_epochs 5 --batch_size 128 \
    --blr 1e-4 --layer_decay 1 --beta2 0.95 --weight_decay 0.3 --drop_path 0.1 \
    --ema 0.9999 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --dist_eval --data_path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR}
