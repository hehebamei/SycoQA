#!/bin/bash

export PROJECT_PATH="/data/shangguanzx/project/ISF-master"
export CUDA_VISIBLE_DEVICES="7"

model_name="Llama-3-8B-Instruct"
dataset_list=(mgsmqq)

for i in ${dataset_list[*]}; do
    python Ha_main.py --model_name $model_name \
                        --dataset "$i" \
                        --print_model_parameter \
                        --save_output \
                        --save_hidden_states
done
