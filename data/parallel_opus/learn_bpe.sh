#!/bin/bash
# Learn Shared BPE from training data
merge_ops=64000

tgtlang=$1
mkdir "en-${1}"

echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
cat "parallel_opus_tokenised.en" "parallel_opus_tokenised.${1}" | \
../subword-nmt/subword_nmt/learn_bpe.py -s $merge_ops > "en-${1}/bpe.${merge_ops}"

# Apply BPE to all tokenised files
echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
for lang in en $1
do
        outfile="en-${tgtlang}/train.$lang"
        ../subword-nmt/subword_nmt/apply_bpe.py -c "en-${tgtlang}/bpe.${merge_ops}" < "parallel_opus_tokenised.${lang}" > "${outfile}"
        echo ${outfile}
done

# Create vocabulary file for BPE
echo "Create vocabulary file for BPE..."
cat "en-${tgtlang}/train.en" "en-${tgtlang}/train.${tgtlang}" | \
../subword-nmt/subword_nmt/get_vocab.py | cut -f1 -d ' ' > "en-${tgtlang}/vocab.bpe"

../subword-nmt/subword_nmt/apply_bpe.py -c "en-${tgtlang}/bpe.${merge_ops}" < "../flores/dev/en_tokenised.dev" > "en-${tgtlang}/dev.en"
../subword-nmt/subword_nmt/apply_bpe.py -c "en-${tgtlang}/bpe.${merge_ops}" < "../flores/devtest/en_tokenised.devtest" > "en-${tgtlang}/devtest.en"

../subword-nmt/subword_nmt/apply_bpe.py -c "en-${tgtlang}/bpe.${merge_ops}" < "../flores/dev/${tgtlang}_tokenised.dev" > "en-${tgtlang}/dev.${tgtlang}"
../subword-nmt/subword_nmt/apply_bpe.py -c "en-${tgtlang}/bpe.${merge_ops}" < "../flores/devtest/${tgtlang}_tokenised.devtest" > "en-${tgtlang}/devtest.${tgtlang}"
