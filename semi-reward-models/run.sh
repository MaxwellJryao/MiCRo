#!/bin/bash

# define parameters
# n_samples=(15 50 100 1000 3000 5000)
n_samples=(10000)
# base_models=("Ray2333/GRM-Llama3.2-3B-rewardmodel-ft" "Ray2333/Gemma-2B-rewardmodel-baseline")
base_models=("Ray2333/GRM-Llama3.2-3B-rewardmodel-ft")

# base output directory
base_output_dir="./test_results"
gradient_accumulation_steps=4

# loop through all parameter combinations
for model in "${base_models[@]}"; do
  for samples in "${n_samples[@]}"; do
    output_dir="${base_output_dir}/$(echo $model | tr '/' '_')/samples_${samples}"
    mkdir -p $output_dir
    
    echo "Running with base_model=$model, n_samples_per_attribute=$samples, output_result_dir=$output_dir"

    # execute script command
    CUDA_VISIBLE_DEVICES=2,3,8,9 accelerate launch --config_file ../configs/config.yaml \
      --main_process_port=29505 --num_processes=4 --gradient_accumulation_steps=$gradient_accumulation_steps learn_teacher.py --num_train_epochs 10 \
      --output_result_dir "$output_dir" \
      --n_samples_per_attribute "$samples" \
      --dataset_dir "./dataset" \
      --labeling_threshold 0.8 \
      --gradient_accumulation_steps=$gradient_accumulation_steps
  done
done
