#!/bin/bash
set -e
BASE_MODEL="meta-llama/Llama-2-7b-hf"
DATA_PATH="pissa-dataset"
MASTER_PORT=41322
TRAIN_GPUS="localhost:0,1"
EVAL_GPU=0
OUTPUT_ROOT="output/code"

RANK=16
LORA_ALPHA=16
DROPOUT=0.3
per_device_train_batch_size=1
gradient_accumulation_steps=1
LR=0.0003
TOTAL_BATCH_SIZE=2

OUTPUT_name=rank-${RANK}-alpha-${LORA_ALPHA}-batch-${TOTAL_BATCH_SIZE}-${per_device_train_batch_size}-${gradient_accumulation_steps}-LR-${LR}-dropout-${DROPOUT}
OUTPUT_PATH=${OUTPUT_ROOT}/${OUTPUT_name}

deepspeed --master_port=${MASTER_PORT} --include=${TRAIN_GPUS} train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank $RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $DROPOUT \
    --data_path $DATA_PATH \
    --sub_task python:50000 \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --merge True

CUDA_VISIBLE_DEVICES=${EVAL_GPU} python utils/gen_vllm.py \
    --model $OUTPUT_PATH \
    --sub_task python \
    --output_file $OUTPUT_PATH/python_response.jsonl
python utils/code_process.py --path $OUTPUT_PATH/python_response.jsonl
evalplus.evaluate --dataset humaneval --samples $OUTPUT_PATH/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $OUTPUT_PATH/mbpp.jsonl
