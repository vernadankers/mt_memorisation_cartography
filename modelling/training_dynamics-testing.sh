#!/bin/bash

DIR_NAME="training_dynamics/seed=1"
trglang=$1

for CHECKPOINT in $SLURM_ARRAY_TASK_ID
do
    # Get likelihood of references for training corpus
    python ../fairseq/fairseq_cli/generate.py en-${trglang}/data/${DIR_NAME}/data-bin \
        --gen-subset train -t $trglang -s 'en' \
        --path en-${trglang}/models/${DIR_NAME}/checkpoint${CHECKPOINT}.pt \
        --max-tokens 2048 \
        --score-reference > en-${trglang}/data/${DIR_NAME}/train${CHECKPOINT}_ref.out
    wait
    # Get hypotheses for training corpus
    python ../fairseq/fairseq_cli/generate.py en-${trglang}/data/${DIR_NAME}/data-bin \
        --gen-subset train -t $trglang -s 'en' \
        --path en-${trglang}/models/${DIR_NAME}/checkpoint${CHECKPOINT}.pt \
        --max-tokens 2048 \
        --beam 5 > en-${trglang}/data/${DIR_NAME}/train${CHECKPOINT}_hyp.out
    wait
done
