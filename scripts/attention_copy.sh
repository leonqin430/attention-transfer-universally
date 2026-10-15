#!/bin/bash
# Attention Copy: the student uses the teacher's attention maps (frozen), all other weights are trained.
# Standard architecture: --model dual_vit_small_patch16 (or dual_vit_tiny_patch16 / dual_vit_base_patch16)
# Native architecture:   --model dual_vit_dinov2_small_patch16 | dual_vit_dinov2_base_patch16 |
#                                dual_vit_clip_base_patch16 | dual_vit_beitv2_base_patch16
# Batch size 128 x 8 GPUs = 1024.
IMAGENET_PATH=${IMAGENET_PATH:-/path/to/imagenet}
TEACHER=${TEACHER:-/path/to/teacher_checkpoint.pth}
MODEL=${MODEL:-dual_vit_small_patch16}
OUTPUT_DIR=${OUTPUT_DIR:-./output_dir/copy_${MODEL}}

torchrun --nproc_per_node=8 main_finetune.py \
    --model ${MODEL} --mode copy --end_layer 0 \
    --finetune ${TEACHER} \
    --epochs 20 --warmup_epochs 5 --batch_size 128 \
    --blr 1e-3 --layer_decay 0.75 --beta2 0.999 --weight_decay 0.05 --drop_path 0 \
    --ema 0.9999 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 \
    --dist_eval --data_path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR}
