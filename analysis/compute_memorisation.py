import argparse
import pickle
from collections import defaultdict, Counter
import os
from sacrebleu.metrics import BLEU
import numpy as np
import scipy.stats


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


def load_scores(corpus_path, pred_path, scores, use_hyp, verbose=True):
    """
    Given a corpus & model predictions, compute likelihood / BLEU for all
    examples in this dataset.
    Args:
        - corpus_path (str): filename that contains train / test data
        - pred_path (str): filename of model predictions
        - scores (dict): dict to which to add the scores for this model
        - use_hyp (bool): whether to compute scores using BLEU
        - verbose (bool): whether to report on progress
    Returns:
        updated dictionary with per-example likelihood / BLEU
    """
    not_found = []
    dataset = reorder_mt(corpus_path + ".en", pred_path, corpus_path + f".{trglang}")
    bleu = BLEU(effective_order=True)
    if verbose and not dataset:
        print(f"{filename} empty.")
    for src, hyp, trg, score in dataset:
        # examples that are empty / too long have default value 1000
        if score is None or float(score) == 1000:
            not_found.append((src, trg))
            continue
        if not use_hyp:
            # log probs to probs (Fairseq uses log2 base)
            score = 2 ** (float(score))
        else:
            score = bleu.sentence_score(
                hyp.replace("@@ ", ""), [trg.replace("@@ ", "")]
            ).score
        scores[(src, trg)].append(score)
    if verbose:
        print(f"{pred_path} had {len(not_found)} not found...")
    return scores


def compute_memorisation(
    start=1,
    stop=41,
    use_hyp=False,
    epoch=100,
    prefix="../modelling/acl_opus/data",
    verbose=False,
    postfix="ref",
    trglang="nl",
):
    """
    Compute counterfactual memorisation as per Feldman et al. (2020)
    Args:
        start (int): model id to start at
        stop (int): model id to stop at
        use_hyp (bool): whether to measure success with BLEU
        epoch (int): epoch from which we want to use predictions
        prefix (str): folder name
        verbose (bool): whether to print progress to the screen
        postfix (str): ref | hyp
        trglang (str): target language, de | nl | fr | es | it
    Returns:
        memorisation (dict): counterfact. mem. per src-trg pair
        train_scores (dict): all train scores from model runs for src-trg pairs
        test_scores (dict): all test scores from model runs for src-trg pairs
    """

    train_scores, test_scores = defaultdict(list), defaultdict(list)
    for i in range(start, stop + 1):
        train_src = f"{prefix}seed={i}/train"
        train_file = f"{prefix}seed={i}/train{epoch}_{postfix}.out"
        test_src = f"{prefix}seed={i}/test"
        test_file = f"{prefix}seed={i}/test{epoch}_{postfix}.out"

        train_scores = load_scores(
            train_src, train_file, train_scores, use_hyp, verbose=verbose
        )
        test_scores = load_scores(
            test_src, test_file, test_scores, use_hyp, verbose=verbose
        )
        if verbose:
            print(
                f"Processing model {i}, {len(train_scores)} train scores"
                + f", {len(test_scores)} test scores."
            )
    return train_scores, test_scores


def finalise_scores(train_scores, test_scores):
    # memorisation scores are train - test likelihood of an example
    memorisation = {
        x: np.mean(train_scores[x]) - np.mean(test_scores[x])
        for x in train_scores
        if x in test_scores
    }
    train_scores = {x: np.mean(y) for x, y in train_scores.items()}
    test_scores = {x: np.mean(y) for x, y in test_scores.items()}
    return memorisation, train_scores, test_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trglang", type=str)
    parser.add_argument("--use_hyp", action="store_true")
    parser.add_argument("--postfix", type=str, default="ref")
    parser.add_argument("--comment", type=str, default="")
    args = parser.parse_args()
    trglang = args.trglang
    postfix = args.postfix
    prefix = f"../modelling/en-{trglang}/data/memorisation_training/"
    save = f"en-{trglang}"

    train_scores, test_scores = compute_memorisation(
        1,
        40,
        prefix=prefix,
        epoch=100,
        trglang=trglang,
        use_hyp=True if postfix == "hyp" else False,
        postfix=postfix,
        verbose=True,
    )

    cm, train_scores, test_scores = finalise_scores(train_scores, test_scores)
    pickle.dump(
        (cm, train_scores, test_scores),
        open(
            f"memorisation_pickled/{save}/counterfactual_memorisation_{postfix}{args.comment}.pickle",
            "wb",
        ),
    )
