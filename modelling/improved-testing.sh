#!/bin/bash

trglang=$1
DIR_NAME="improved_training/seed=${SLURM_ARRAY_TASK_ID}_${2}"

for CHECKPOINT in _best
do
    # Use devtest because we used dev in sec6.1
    python ../fairseq/fairseq_cli/preprocess.py --srcdict en-${trglang}/data/${DIR_NAME}/data-bin/dict.en.txt \
        --tgtdict en-${trglang}/data/${DIR_NAME}/data-bin/dict.${trglang}.txt \
        --testpref ../data/parallel_opus/en-${trglang}/devtest \
        --destdir en-${trglang}/data/${DIR_NAME}/data-bin \
        -s en -t $trglang

    # Get the probabilities of the Flores reference translation
    python ../fairseq/fairseq_cli/generate.py en-${trglang}/data/${DIR_NAME}/data-bin \
        --gen-subset test -s 'en' -t $trglang \
        --path en-${trglang}/models/${DIR_NAME}/checkpoint${CHECKPOINT}.pt \
        --batch-size 64 --score-reference > en-${trglang}/data/${DIR_NAME}/flores-devtest${CHECKPOINT}_ref.out
    wait

    # Flores devtest translations with beam size of 5
    python ../fairseq/fairseq_cli/generate.py en-${trglang}/data/${DIR_NAME}/data-bin \
        --gen-subset test -s 'en' -t $trglang \
        --path en-${trglang}/models/${DIR_NAME}/checkpoint${CHECKPOINT}.pt \
        --batch-size 64 --beam 5 > en-${trglang}/data/${DIR_NAME}/flores-devtest${CHECKPOINT}_hyp.out
    wait

    # Try to elicit Flores hallucinations
    python ../fairseq/fairseq_cli/interactive.py en-${trglang}/data/${DIR_NAME}/data-bin \
        --input "../data/parallel_opus/en-nl/dev_hallucinations.en" \
        -s 'en' --path en-${trglang}/models/${DIR_NAME}/checkpoint${CHECKPOINT}.pt \
        --batch-size 64 --beam 1 --buffer-size 64 \
        > en-${trglang}/data/${DIR_NAME}/dev_hallucinations_model${CHECKPOINT}.out
    wait
done
