from collections import defaultdict, Counter
import os
import numpy as np
import random
from scipy.spatial import KDTree
import pickle
import tqdm
import argparse
from sacrebleu.metrics import CHRF, BLEU
from sacremoses import MosesDetokenizer


def reorder_simple(lines, letter="D"):
    """
    Reorder lines, since Fairseq shuffles them while translating.
    Order based on the index indicated after "D-...", "S-..." or "H-...".
    
    Args:
        - lines: list of str
    Returns:
        - list of str
    """
    sentences = []
    for line in lines:
        line = line.split("\t")
        if f"{letter}-" in line[0]:
            index = int(line[0].split('-')[1])
            sentence = line[2].strip().replace("@@ ", "")
            sentences.append((index, sentence))
    _, sentences = zip(*sorted(sentences))
    return sentences


def compute_bleu(tgt, hyp, lang='nl'):
    md = MosesDetokenizer(lang)
    bleu = BLEU(force=True)
    hyp = [md.detokenize(x.replace("@@ ", "").split()) for x in hyp]
    tgt = [md.detokenize(x.replace("@@ ", "").split()) for x in tgt]
    bleu_score = bleu.corpus_score(hyp, [list(tgt)]).score
    return bleu_score


def compute_probs(lines):
    return np.mean([2**float(x.split("\t")[3]) for x in lines])


def reorder_for_probs(filename, letter="P"):
    """
    Reorder lines, since Fairseq shuffles them while translating.
    Order based on the index indicated after "D-...", "S-..." or "H-...".
    
    Args:
        - lines: list of str
    Returns:
        - list of str
    """
    sentences = []
    for line in open(filename).readlines():
        line = line.split("\t")
        if f"{letter}-" in line[0]:
            index = int(line[0].split('-')[1])
            probs = line[1].strip().split()
            probs = [2**float(x) for x in probs]
            sentences.append((index, probs))
    _, probs = zip(*sorted(sentences))
    return probs


def reorder_mt(src, out, trg):
    """
    Reorder predictions obtained from fairseq, combine them with src & trg.
    Args:
        - src (str): filename of source sequences
        - out (str): filename of hypotheses
        - trg (str): filename of target sequences
    Returns:
        list with tuples that contain (source, hypothesis, target, likelihood)
    """
    src_lines = [s.strip() for s in open(src, encoding="utf-8").readlines()]
    trg_lines = [t.strip() for t in open(trg, encoding="utf-8").readlines()]
    out_lines = [o.strip() for o in open(out, encoding="utf-8").readlines()]

    hypotheses = dict()
    for line in out_lines:
        if "H-" in line[:2]:
            idx = int(line.split("\t")[0].replace("H-", ""))
            score = float(line.split("\t")[1])
            try:
                hyp = line.split("\t")[2]
            except:
                hyp = score
                score = 1000
            hypotheses[idx] = (hyp, score)

    corpus = []
    for i, (s, t) in enumerate(zip(src_lines, trg_lines)):
        hyp, score = hypotheses[i]
        corpus.append((s, hyp, t, score))
    return corpus
