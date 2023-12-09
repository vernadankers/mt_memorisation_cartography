#!/bin/bash

seed=$1
trglang=$2
postfix=$3

DIR_NAME="performance_impact/subset=${SLURM_ARRAY_TASK_ID}_${postfix}_seed=${seed}"

fairseq-preprocess --source-lang en --target-lang $trglang \
    --trainpref ../analysis/subsets/subset=${SLURM_ARRAY_TASK_ID}_${postfix} \
    --testpref ../data/parallel_opus/en-${trglang}/devtest \
    --validpref ../data/parallel_opus/en-${trglang}/dev \
    --destdir en-${trglang}/data/${DIR_NAME}/data-bin \
    --seed $seed --joined-dictionary

mkdir en-${trglang}/models/${DIR_NAME}

python ../fairseq/fairseq_cli/train.py \
    en-${trglang}/data/${DIR_NAME}/data-bin \
    --arch transformer_regular \
    --save-dir en-${trglang}/models/${DIR_NAME} --share-all-embeddings \
    --fp16 --max-update 200000 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 10000 --update-freq 2 \
    --max-epoch 50 \
    --seed $seed --validate-interval 5 \
    --eval-bleu --eval-bleu-args '{"beam": 5}' --eval-bleu-remove-bpe
done
