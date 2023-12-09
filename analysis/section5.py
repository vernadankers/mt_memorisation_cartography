import math
import scipy.stats
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from map_features import Example
from utils import reorder_mt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def process_proxies(lang):
    """
    Process reference scores and hypothesis scores over 50 epochs into
    training dynamics signals, that are proxies for memorisation metrics.
    Args:
        - lang (str): de | nl | fr | it | es
    """
    probs = defaultdict(list)

    # To link proxies to examples, we need tokenised and untokenised data
    src_fn = f"../data/parallel_opus/en-{lang}/train.en"
    trg_fn = f"../data/parallel_opus/en-{lang}/train.{lang}"
    src_fn_untok = "../data/parallel_opus/parallel_opus.en"
    trg_fn_untok = f"../data/parallel_opus/en-{lang}/parallel_opus.{lang}"
    src_untok = [l.strip()
                 for l in open(src_fn_untok, encoding="utf-8").readlines()]
    trg_untok = [l.strip()
                 for l in open(trg_fn_untok, encoding="utf-8").readlines()]

    # Record reference probabilities for epochs 1 to 50
    for epoch in range(1, 51):
        hyp_fn = (
            f"../modelling/en-{lang}/data/training_dynamics/seed=1/train{epoch}_ref.out"
        )
        lines = reorder_mt(src_fn, hyp_fn, trg_fn)

        assert len(lines) == len(src_untok)
        for line, s_untok, t_untok in zip(lines, src_untok, trg_untok):
            src, hyp, trg, prob = line
            if prob is None or prob == -1000:
                continue
            probs[(src, trg, s_untok, t_untok)].append(2 ** float(prob))

    # Record hypothesis probabilities for epoch 50
    probs_hyp = defaultdict(list)
    hyp_fn = f"../modelling/en-{lang}/data/training_dynamics/seed=1/train50_hyp.out"
    lines = reorder_mt(src_fn, hyp_fn, trg_fn)
    assert len(lines) == len(src_untok)
    for line, s_untok, t_untok in zip(lines, src_untok, trg_untok):
        src, hyp, trg, prob = line
        probs_hyp[(src, trg, s_untok, t_untok)] = 2 ** float(prob)

    proxies = dict()
    for example in probs:
        # Sum the decreases in the likelihood, this is our forgetting metric
        t = 0
        previous_p = None
        for p in probs[example]:
            if previous_p is not None and p < previous_p:
                t += previous_p - p
            previous_p = p

        proxies[example] = (
            np.mean(probs[example]),
            np.std(probs[example]),
            probs[example][-1] - np.mean(probs[example]),
            t,
            probs_hyp[example],
            probs[example][-1],
        )
    pickle.dump(proxies, open(f"proxies_{lang}.pickle", "wb"))


def visualise_correlations_proxies_metrics(lang):
    """
    Visualise heatmap of 3 metrics x 6 proxies with Pearson correlations.
    Args:
        - lang (str): nl | de | fr | it | es
    """
    proxies = pickle.load(
        open(f"memorisation_pickled/en-{lang}/proxies_{lang}.pickle", "rb")
    )
    examples = pickle.load(
        open(f"memorisation_pickled/en-{lang}/examples_{lang}.pickle", "rb")
    )

    heatmap = np.zeros((3, 6))
    for i in range(6):
        x, y = [], []
        for s in proxies:
            x.append(proxies[s][i])
            y.append(examples[s].train_likelihood_ref)
        heatmap[0, i] = scipy.stats.pearsonr(x, y)[0]

        x, y = [], []
        for s in proxies:
            x.append(proxies[s][i])
            y.append(examples[s].test_likelihood_ref)
        heatmap[1, i] = scipy.stats.pearsonr(x, y)[0]

        x, y = [], []
        for s in proxies:
            x.append(proxies[s][i])
            y.append(max(0, examples[s].counterfactual_memorisation_ref))
        heatmap[2, i] = scipy.stats.pearsonr(x, y)[0]
    plt.figure(figsize=(8, 3.3))
    ax = sns.heatmap(
        heatmap,
        cmap="RdYlBu_r",
        annot=True,
        vmin=-0.5,
        vmax=0.5,
        cbar_kws={"label": r"Pearson's $r$"},
        annot_kws={"fontsize": 14},
    )
    ax.set_xticklabels(
        [
            "confidence",
            "variability",
            "LL 50 - conf",
            "forgetting",
            "hyp. LL 50",
            "LL 50",
        ],
        rotation=30,
    )
    ax.set_yticklabels(
        [
            "training memorisation",
            "generalisation score",
            "counterfactual memorisation",
        ],
        rotation=0,
    )
    plt.savefig(
        f"appendix_figures/signals_correlations_{lang}.pdf", bbox_inches="tight"
    )
    print("Saved visualisation of correlations signals<->metrics to file.")


def load_data(threshold):
    """
    Load all examples and training dynamics proxies from file.

    Args:
        - threshold (float): ratio of data to load (when low on memory)
    Returns:
        - examples dict: keys are languages, values are pickled dicts
        - proxies dict: keys are languages, values are pickled dicts
    """

    def subset(dict_, threshold):
        return {k: dict_[k] for k in dict_ if random.random() < threshold}

    def subset2(dict_, keys):
        return {k: dict_[k] for k in keys}

    t = threshold
    examples = dict()
    examples["de"] = subset(
        pickle.load(
            open("memorisation_pickled/en-de/examples_de.pickle", "rb")), t
    )
    examples["nl"] = subset(
        pickle.load(
            open("memorisation_pickled/en-nl/examples_nl.pickle", "rb")), t
    )
    examples["it"] = subset(
        pickle.load(
            open("memorisation_pickled/en-it/examples_it.pickle", "rb")), t
    )
    examples["fr"] = subset(
        pickle.load(
            open("memorisation_pickled/en-fr/examples_fr.pickle", "rb")), t
    )
    examples["es"] = subset(
        pickle.load(
            open("memorisation_pickled/en-es/examples_es.pickle", "rb")), t
    )
    print(f"Loaded memorisation metrics with threshold {threshold}.")

    proxies = dict()
    proxies["de"] = subset2(
        pickle.load(open("memorisation_pickled/en-de/proxies_de.pickle", "rb")),
        examples["de"].keys(),
    )
    proxies["nl"] = subset2(
        pickle.load(open("memorisation_pickled/en-nl/proxies_nl.pickle", "rb")),
        examples["nl"].keys(),
    )
    proxies["it"] = subset2(
        pickle.load(open("memorisation_pickled/en-it/proxies_it.pickle", "rb")),
        examples["it"].keys(),
    )
    proxies["es"] = subset2(
        pickle.load(open("memorisation_pickled/en-es/proxies_es.pickle", "rb")),
        examples["es"].keys(),
    )
    proxies["fr"] = subset2(
        pickle.load(open("memorisation_pickled/en-fr/proxies_fr.pickle", "rb")),
        examples["fr"].keys(),
    )
    print(f"Loaded proxies with threshold {threshold}.")
    return examples, proxies


def process_data(examples, proxies=None):
    """
    Process dictionaries of examples / training signals into matrices.
    Args:
        - examples (dict): langs as keys and Example dicts as values
        - proxies (dict): langs as keys and dictionaries as values, mapping
          src-trg pairs to a list of training signals
    Returns:
        - X: matrix of num_examples x 28 or 28 + 6
        - y: vector of num_examples x 3
    """
    X, y = [], []
    for key in examples.keys():
        sample = []
        for f in examples[key].numerical_features:
            sample.append(examples[key].numerical_features[f])
        if proxies is not None:
            sample.extend(proxies[key])
        X.append(sample)
        y_sample = []
        for metric in [
                "train_likelihood_ref",
                "test_likelihood_ref",
                "counterfactual_memorisation_ref",
        ]:
            y_sample.append(getattr(examples[key], metric))
        y.append(y_sample)
    return X, y


def l1(x, y):
    return np.mean([abs(a - b) for a, b in zip(x, y)])


def train_mlps(examples, proxies, use_proxies=False):
    """
    Train MLPRegressor to predict memorisation metrics, collect results
    in matrices.
    Args:
        - examples (dict): langs as keys and Example dicts as values
        - proxies (dict): langs as keys and dictionaries as values, mapping
          src-trg pairs to a list of training signals
        - use_proxies (bool): whether to include the proxies during training
    Returns:
        - matrix 5x5 with correlation results
        - matrix 5x5 with abs difference results
    """
    matrix_cor = defaultdict(lambda: np.zeros((5, 5)))
    matrix_l1 = defaultdict(lambda: np.zeros((5, 5)))
    for i, train_language in enumerate(["de", "nl", "fr", "es", "it"]):
        model = MLPRegressor(hidden_layer_sizes=(
            100, 100), max_iter=20, verbose=False)
        X, y = process_data(
            examples[train_language], proxies[train_language] if use_proxies else None
        )
        print(f"Training MLP on {train_language}...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)
        model.fit(X_train, y_train)

        for j, test_language in enumerate(["de", "nl", "fr", "es", "it"]):
            print(f"    - Testing on {test_language}...")
            X2, y2 = process_data(
                examples[test_language], proxies[test_language] if use_proxies else None
            )
            if train_language == test_language:
                train_ll_t, test_ll_t, cm_ll_t = zip(*y_test)
                train_ll_p, test_ll_p, cm_ll_p = zip(*model.predict(X_test))
            else:
                train_ll_t, test_ll_t, cm_ll_t = zip(*y2)
                train_ll_p, test_ll_p, cm_ll_p = zip(*model.predict(X2))

            matrix_cor["train_ll"][i, j] = scipy.stats.pearsonr(train_ll_t, train_ll_p)[0]
            matrix_cor["test_ll"][i, j] = scipy.stats.pearsonr(test_ll_t, test_ll_p)[0]
            matrix_cor["cm_ll"][i, j] = scipy.stats.pearsonr(cm_ll_t, cm_ll_p)[0]
            matrix_l1["train_ll"][i, j] = l1(train_ll_t, train_ll_p)
            matrix_l1["test_ll"][i, j] = l1(test_ll_t, test_ll_p)
            matrix_l1["cm_ll"][i, j] = l1(cm_ll_t, cm_ll_p)
    return matrix_cor, matrix_l1


def train_and_visualise(examples, proxies):
    """
    Train MLPs to predict mem. metrics, and apply across languages.
    Args:
        - examples (dict): langs as keys and Example dicts as values
        - proxies (dict): langs as keys and dictionaries as values, mapping
          src-trg pairs to a list of training signals
    """
    # Now train MLPs on one language and test on another
    matrix_cor1, matrix_l11 = train_mlps(examples, proxies, use_proxies=False)
    matrix_cor2, matrix_l12 = train_mlps(examples, proxies, use_proxies=True)

    def barplot(matrix, ylabel, legend, filename, ymin, ymax):
        metric_names = ["training mem.", "gen. score", "counterfactual mem."]
        metrics = ["train_ll", "test_ll", "cm_ll"]
        plt.figure(figsize=(3.7, 2.7))
        x, y, h = [], [], []
        for metric, name in zip(metrics, metric_names):
            x.extend(["de", "nl", "fr", "es", "it"])
            y.extend(matrix[metric][0].tolist())
            h.extend([name] * 5)

        ax = sns.barplot(
            x=x,
            y=y,
            hue=h,
            palette=sns.color_palette("Spectral_r", 8),
            saturation=1,
            linewidth=1,
            edgecolor="black",
        )
        if legend:
            plt.legend(fontsize=13, frameon=False)
        else:
            plt.legend([], [], frameon=False)
        sns.despine(top=True, right=True)
        plt.ylabel(ylabel)
        ax.set_ylim(ymin, ymax)
        plt.savefig(filename, bbox_inches="tight")

    # Viusalise correlation for training on features only
    barplot(matrix_cor1, r"Pearson's $r$", True,
            "figures/features_cor.pdf", 0.5, 1.0)
    # Visualise abs difference for training on features only
    barplot(
        matrix_l11, "absolute difference", False, "figures/features_l1.pdf", 0, 0.15
    )
    # Viusalise correlation for training on features & training signals
    barplot(
        matrix_cor2,
        r"Pearson's $r$",
        False,
        "figures/features_signals_cor.pdf",
        0.5,
        1.0,
    )
    # Viusalise abs difference for training on features & training signals
    barplot(
        matrix_l12,
        "absolute difference",
        False,
        "figures/features_signals_l1.pdf",
        0,
        0.15,
    )
    print("Visualised results for training on DE in barplots, saved to file.")

    # Visualise training and testing across languages for all languages,
    # CM-LL only as an example
    plt.figure(figsize=(8, 4))
    ax = sns.heatmap(
        matrix_l12["cm_ll"],
        cmap="Blues",
        annot=True,
        cbar_kws={"label": r"absolute difference"},
        annot_kws={"fontsize": 14},
    )
    ax.set_xticklabels(["de", "nl", "fr", "es", "it"])
    ax.set_yticklabels(["de", "nl", "fr", "es", "it"], rotation=0)
    plt.xlabel("test on language x")
    plt.ylabel("train on language y")
    plt.savefig("appendix_figures/cm_ll_predictions.pdf", bbox_inches="tight")
    print("Visualised results for training on all in heatmaps, saved to file.")


if __name__ == "__main__":
    sns.set_context("talk")
    # process_proxies(lang)

    ### Appendix B: correlations metrics <-> signals
    visualise_correlations_proxies_metrics("nl")

    ### Main paper: train on DE and test on all, + appendix, train/test on all
    # First load a subset of the data when working with low memory
    threshold = 1
    examples, proxies = load_data(threshold)
    # Second, train MLPs, apply across languages
    train_and_visualise(examples, proxies)
