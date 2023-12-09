#!/bin/bash

SEED=$SLURM_ARRAY_TASK_ID
trglang=$1

DIR_NAME="memorisation_training/seed=${SEED}"

# Target likelihood and predicted translations for 1M OPUS sents
for CHECKPOINT in 100
do
    for subset in train test
    do
        # Score reference translation
        python ../fairseq/fairseq_cli/generate.py en-${trglang}/data/${DIR_NAME}/data-bin \
        --gen-subset $subset -s 'en' -t $trglang \
        --path en-${trglang}/models/${DIR_NAME}/checkpoint${CHECKPOINT}.pt \
        --max-tokens 2048 --score-reference  > en-${trglang}/data/${DIR_NAME}/${subset}${CHECKPOINT}_ref.out
        wait

        # Generate hyps
        python ../fairseq/fairseq_cli/generate.py en-${trglang}/data/${DIR_NAME}/data-bin \
        --gen-subset $subset -s 'en' -t $trglang \
        --path en-${trglang}/models/${DIR_NAME}/checkpoint${CHECKPOINT}.pt  \
        --max-tokens 2048 --beam 1 > en-${trglang}/data/${DIR_NAME}/${subset}${CHECKPOINT}_hyp.out
        wait
    done
done