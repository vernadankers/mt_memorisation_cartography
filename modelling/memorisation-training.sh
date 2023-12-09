#!/bin/bash

trglang=$1
SEED=$SLURM_ARRAY_TASK_ID
DIR_NAME="memorisation_training/seed=${SEED}"

python subsample.py --seed $SLURM_ARRAY_TASK_ID \
    --trainpref ../data/parallel_opus/en-${trglang}/train \
    --testpref ../data/parallel_opus/en-${trglang}/test \
    --destdir en-${trglang}/data/${DIR_NAME} --src en \
    --trg ${trglang} --ratio 0.5

fairseq-preprocess --source-lang en --target-lang ${trglang} \
    --trainpref en-${trglang}/data/${DIR_NAME}/train \
    --validpref ../data/parallel_opus/en-${trglang}/dev \
    --testpref en-${trglang}/data/${DIR_NAME}/test \
    --destdir en-${trglang}/data/${DIR_NAME}/data-bin --seed $SEED --joined-dictionary

python ../fairseq/fairseq_cli/train.py \
    en-${trglang}/data/memorisation_training/seed=${SEED}/data-bin \
    --arch transformer_regular \
    --save-dir en-${trglang}/models/${DIR_NAME} --share-all-embeddings \
    --fp16 --max-update 200000 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 10000 --update-freq 2 \
    --save-interval 10 --max-epoch 100 \
    --seed $SLURM_ARRAY_TASK_ID --validate-interval 5 \
    --eval-bleu --eval-bleu-args '{"beam": 5}' --eval-bleu-remove-bpe
