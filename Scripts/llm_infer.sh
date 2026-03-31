#!/bin/bash

export PROJECT_PATH=""

model_name="Mistral-7B-Instruct-v0.3"
dataset_list=()

for i in ${dataset_list[*]}; do
    python Ha_main.py --model_name $model_name \
                        --dataset "$i" \
                        --print_model_parameter \
                        --save_output \
                        --save_hidden_states
done

