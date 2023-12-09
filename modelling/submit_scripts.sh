#!/bin/bash

setting=$1
trglang=$2

if [ "$setting" = "memorisation-training" ]; then
    echo $setting
    sbatch --array=1-40 memorisation-training.sh $trglang 

elif [ "$setting" = "memorisation-testing" ]; then
    echo $setting
    sbatch --array=1-40 memorisation-testing.sh $trglang

elif [ "$setting" = "training-dynamics-training" ]; then
    echo $setting
    sbatch training_dynamics-training.sh $trglang

elif [ "$setting" = "training-dynamics-testing" ]; then
    echo $setting
    sbatch --array=1-50 training_dynamics-testing.sh $trglang

elif [ "$setting" = "performance-training" ]; then
    echo $setting
    postfix=$3
    for seed in 1 2 3
    do
        sbatch --array=0-54 performance_impact-training.sh $seed $trglang $postfix
    done

elif [ "$setting" = "performance-testing" ]; then
    echo $setting
    postfix=$3
    for seed in 1 2 3
    do
        sbatch --array=0-54 performance_impact-testing.sh $seed $trglang $postfix
    done

elif [ "$setting" = "improved-training" ]; then
    echo $setting
    for setup in random bleu_hal log_probability
    do
        sbatch --array=1-3 improved-training.sh $trglang $setup
    done

elif [ "$setting" = "improved-testing" ]; then
    echo $setting
    for setup in random bleu_hal log_probability
    do
        sbatch --array=1-3 improved-testing.sh $trglang $setup
    done

fi
