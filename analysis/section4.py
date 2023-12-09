import math
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import pandas as pd
from map_features import *
from collections import defaultdict, Counter


def load_with_threshold(threshold=1.01):
    """
    Load 5M datapoints, potentially with a threshold when not having
    enough memory to load everything in memory:
    Args:
        - threshold (float): 0 - 1, 0M to 5M
    Returns:
        - dict of custom Example objects as values and src-trg keys
    """
    # you can set the threshold lower if there's not enough memory to load all data at once
    examples = dict()
    for language in ["nl", "de", "fr", "es", "it"]:
        examples.update(
            {
                k: v
                for k, v in pickle.load(
                    open(
                        f"../../memorisation_mt/analysis_/memorisation_pickled/en-{language}/examples_{language}.pickle",
                        "rb",
                    )
                ).items()
                if random.random() < threshold
            }
        )
    return examples


def visualise_map(examples, colour_scheme, hue_title, figname):
    """
    Visualise all datapoints as dots, include marginal distributions.
    Args:
        - examples: dict with src-trg pairs as keys, Example obj as values
        - colour_scheme (str(: counterfactual_memorisation_ref|hyp
        - hue_title (str): legend title
        - figname (str): save figure under this name
    """
    x, y, hue = [], [], []
    for e in list(examples.values()):
        if "hyp" in colour_scheme:
            x.append(e.train_likelihood_hyp / 100)
            y.append(e.test_likelihood_hyp / 100)
            hue.append(math.ceil(max(0.1, getattr(e, colour_scheme) / 100) * 10) / 10)
        else:
            x.append(e.train_likelihood_ref)
            y.append(e.test_likelihood_ref)
            hue.append(math.ceil(max(0.1, getattr(e, colour_scheme)) * 10) / 10)

    data = {"training memorisation": x, "generalisation score": y, hue_title: hue}
    grid = sns.jointplot(
        data=data,
        x="training memorisation",
        y="generalisation score",
        hue=hue_title,
        joint_kws={"alpha": 0.05, "s": 50},
        linewidth=0,
        palette="Spectral_r",
        height=5,
        legend="full",
    )
    grid.ax_joint.plot([0, 1], [0, 1], linestyle="--", color="black")
    grid.ax_joint.set_xlim(0, 1)
    grid.ax_joint.set_ylim(0, 1)
    handles, labels = grid.ax_joint.get_legend_handles_labels()
    grid.ax_joint.legend(
        handles,
        labels,
        title="counterfactual\nmemorisation",
        frameon=False,
        bbox_to_anchor=(1.16, 1.1),
        fontsize=14,
        title_fontsize=14,
    )
    plt.savefig(figname, bbox_inches="tight")


def visualise_per_feature(examples):
    """
    Create a density plot per numerical feauture to demonstrate how it
    correlates with training memorisation, generalisation score and
    counterfactual memorisation.
    Args:
        - examples: dict with src-trg pairs as keys and Example objects
          as values. We need the .numerical_features attribute here.
    """
    for feature in list(examples.values())[0].numerical_features.keys():
        print(f"Visualised {feature}...")
        trainlabel = "training memorisation"
        testlabel = "generalisation score"
        cmlabel = "counterfactual mem."
        huelabel = feature.replace("_", " ")

        train, test, cm = [], [], []
        hue = []
        for e in examples.values():
            # Bin hue values differently depending on the feature visualised
            if feature in [
                "src_length",
                "trg_length",
                "src_length_tokenised",
                "trg_length_tokenised",
            ]:
                if e.numerical_features[feature] > 100:
                    e.numerical_features[feature] = 100
                hue.append(round(e.numerical_features[feature] / 10) * 10)
            elif "freq" in feature:
                hue.append(round(e.numerical_features[feature]))
            elif "difference" in feature:
                hue.append(
                    max(-25, min(25, round(e.numerical_features[feature] / 5) * 5))
                )
            else:
                hue.append(round(e.numerical_features[feature], 1))
            train.append(e.train_likelihood_ref)
            test.append(e.test_likelihood_ref)
            cm.append(max(0, e.counterfactual_memorisation_ref))

        fig, [ax1, ax2, ax3] = plt.subplots(
            1, 3, figsize=(10.5, 3), sharex=True, sharey=True
        )
        data = {trainlabel: train, testlabel: test, cmlabel: cm, huelabel: hue}
        sns.kdeplot(
            data=data,
            x=trainlabel,
            hue=huelabel,
            ax=ax1,
            palette="Spectral_r",
            common_norm=False,
            bw_adjust=1,
            legend=False,
            warn_singular=False,
        )
        sns.kdeplot(
            data=data,
            x=testlabel,
            hue=huelabel,
            ax=ax2,
            palette="Spectral_r",
            common_norm=False,
            bw_adjust=1,
            legend=False,
            warn_singular=False,
        )
        sns.kdeplot(
            data=data,
            x=cmlabel,
            hue=huelabel,
            ax=ax3,
            palette="Spectral_r",
            common_norm=False,
            bw_adjust=1,
            legend=True,
            warn_singular=False,
        )
        sns.move_legend(
            ax3,
            loc="upper right",
            ncol=2,
            fontsize=13,
            title_fontsize=13,
            frameon=False,
            columnspacing=0.5,
            labelspacing=0.15,
            bbox_to_anchor=(1.55, 1.15),
            title=huelabel.replace("tokenised", "tok."),
        )

        ax1.set_xlim(0, 1)
        ax2.set_xlim(0, 1)
        ax3.set_xlim(0, 1)
        ax1.set_ylabel("density")
        ax2.set_ylabel("density")
        ax3.set_ylabel("density")
        sns.despine(top=True, right=True)
        plt.tight_layout()
        plt.savefig(f"feature_figures/{feature}.pdf", bbox_inches="tight")


def visualise_correlations_features(examples, filename):
    """
    Visualise correlations BETWEEN individual features.
    Args:
        - examples: dict with src-trg pairs as keys and Example object values
        - filename (str): save figure under this name
    """
    features = list(examples.values())[0].numerical_features.keys()
    heatmap = np.zeros((len(features), len(features)))
    heatmap2 = np.zeros((len(features), len(features)))

    for i, f in enumerate(features):
        print(f"{i} out of {len(features)}")
        for j, f2 in enumerate(features):
            x, y = [], []
            for e in examples.values():
                x.append(e.numerical_features[f])
                y.append(e.numerical_features[f2])
            heatmap[j, i] = scipy.stats.spearmanr(x, y)[0]

    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(
        heatmap,
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        mask=abs(heatmap) < 0.5,
        annot=True,
        alpha=0.75,
        annot_kws={"fontsize": 10},
        cbar_kws={"label": "Spearman's " + r"$\rho$", "pad": 0.01},
    )
    ax.set_xticklabels([f.replace("_", " ") for f in features], rotation=90)
    ax.set_yticklabels([f.replace("_", " ") for f in features], rotation=0)
    plt.savefig(filename, bbox_inches="tight")


def visualise_correlations_metrics(examples, filename, postfix="ref"):
    """
    Visualise correlations between features and metrics.
    Args:
        - examples: dict with src-trg pairs as keys and Example object values
        - filename (str): save figure under this name
        - postfix (str): ref | hyp, LL vs BLEU
    """
    features = list(examples.values())[0].numerical_features.keys()
    heatmap = np.zeros((3, len(features)))

    metrics = [
        f"train_likelihood_{postfix}",
        f"test_likelihood_{postfix}",
        f"counterfactual_memorisation_{postfix}",
    ]
    metric_names = ["TM", "GS", "CM"]
    for i, f in enumerate(features):
        for j, m in enumerate(metrics):
            x, y = [], []
            for e in examples.values():
                x.append(e.numerical_features[f])
                y.append(max(0, getattr(e, m)))
            heatmap[j, i] = scipy.stats.spearmanr(x, y)[0]

    plt.figure(figsize=(16, 2))
    ax = sns.heatmap(
        heatmap,
        cmap="RdYlBu_r",
        vmin=-0.5,
        vmax=0.5,
        annot=True,
        alpha=0.75,
        annot_kws={"fontsize": 10},
        cbar_kws={"label": "Spearman's " + r"$\rho$", "pad": 0.02},
        fmt=".2f",
    )
    ax.set_xticklabels([f.replace("_", " ") for f in features], rotation=90)
    ax.set_yticklabels(metric_names, rotation=0)
    plt.savefig(filename, bbox_inches="tight")


def compute_correlation_LL_BLEU():
    """
    Print Spearman's correlation between likelihood- & BLEU-based metrics.
    """
    examples = {
        k: v
        for k, v in pickle.load(
            open(
                f"memorisation_pickled/en-nl/examples_nl.pickle",
                "rb",
            )
        ).items()
    }

    tm_h, gs_h, cm_h = [], [], []
    tm_r, gs_r, cm_r = [], [], []
    for e in list(examples.values()):
        tm_h.append(e.train_likelihood_hyp / 100)
        gs_h.append(e.test_likelihood_hyp / 100)
        cm_h.append(max(0, e.counterfactual_memorisation_hyp / 100))
        tm_r.append(e.train_likelihood_ref)
        gs_r.append(e.test_likelihood_ref)
        cm_r.append(max(0, e.counterfactual_memorisation_ref))
    print("TM, rho = ", scipy.stats.spearmanr(tm_h, tm_r))
    print("GS, rho = ", scipy.stats.spearmanr(gs_h, gs_r))
    print("CM, rho = ", scipy.stats.spearmanr(cm_h, cm_r))


def compare_languages(examples):
    """
    Compute correlations for TM, GS and CM across the 5 languages.
    Store correlations as heatmaps.
    Args:
        - examples: dict with src-trg keys and Example objects as values
    """
    languages = ["de", "nl", "fr", "es", "it"]
    # Load corpora to ensure we process examples across langs in the same
    # order
    corpora = {
        l: list(
            zip(
                [
                    l.strip()
                    for l in open(
                        f"../data/parallel_opus/en-{l}/train.en", encoding="utf-8"
                    ).readlines()
                ],
                [
                    l.strip()
                    for l in open(
                        f"../data/parallel_opus/en-{l}/train.{l}", encoding="utf-8"
                    ).readlines()
                ],
                [
                    l.strip()
                    for l in open(
                        f"../data/parallel_opus/parallel_opus.en", encoding="utf-8"
                    ).readlines()
                ],
                [
                    l.strip()
                    for l in open(
                        f"../data/parallel_opus/en-{l}/parallel_opus.{l}",
                        encoding="utf-8",
                    ).readlines()
                ],
            )
        )
        for l in languages
    }

    heatmap_train = np.zeros((5, 5))
    heatmap_test = np.zeros((5, 5))
    heatmap_cm = np.zeros((5, 5))

    # Compute stats pairwise per lang
    for i, l1 in enumerate(languages):
        for j, l2 in enumerate(languages):
            print(f"Comparing {l1} to {l2}")
            e1 = [examples[st1] for st1 in corpora[l1]]
            e2 = [examples[st2] for st2 in corpora[l2]]
            train1 = [e.train_likelihood_ref for e in e1]
            test1 = [e.test_likelihood_ref for e in e1]
            cm1 = [e.counterfactual_memorisation_ref for e in e1]
            train2 = [e.train_likelihood_ref for e in e2]
            test2 = [e.test_likelihood_ref for e in e2]
            cm2 = [e.counterfactual_memorisation_ref for e in e2]
            heatmap_train[i, j] = scipy.stats.pearsonr(train1, train2)[0]
            heatmap_test[i, j] = scipy.stats.pearsonr(test1, test2)[0]
            heatmap_cm[i, j] = scipy.stats.pearsonr(cm1, cm2)[0]

    def visualise_correlations(heatmap, filename):
        plt.figure(figsize=(4, 4))
        mask = np.triu(np.ones_like(heatmap, dtype=bool))
        ax = sns.heatmap(
            heatmap,
            cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
            mask=mask,
            annot=True,
            cbar=False,
            cbar_kws={"label": "Pearson's " + r"$r$"},
        )
        ax.set_xticklabels(languages)
        ax.set_yticklabels(languages, rotation=0)
        plt.xlim(0, 4)
        plt.ylim(5, 1)
        plt.savefig(f"appendix_figures/{filename}.pdf", bbox_inches="tight")
        plt.show()

    # Save heatmaps separately
    visualise_correlations(heatmap_train, "language_correlations_train_ll.pdf")
    visualise_correlations(heatmap_test, "language_correlations_test_ll.pdf")
    visualise_correlations(heatmap_cm, "language_correlations_cm_ll.pdf")


if __name__ == "__main__":
    sns.set_context("talk")

    ### Illustrative map for Spanish
    examples = pickle.load(
        open(
            f"memorisation_pickled/en-es/examples_es.pickle",
            "rb",
        )
    )
    visualise_map(
        examples,
        colour_scheme="counterfactual_memorisation_ref",
        hue_title="count. mem.",
        figname="figures/mem_ll_es.png",
    )

    ### Now load examples from all languages
    examples = load_with_threshold(0.1)
    # Density plot per feature
    visualise_per_feature(examples)
    # Correlations among features
    visualise_correlations_features(
        examples, "appendix_figures/correlations_features.pdf"
    )
    # Correlations between metrics and features
    visualise_correlations_metrics(
        examples,
        "appendix_figures/correlations_features-metrics_ref.pdf",
        postfix="ref",
    )
    # Correlations across the 5 languages, per metric
    compare_languages(examples)

    ### Correlation hyp-based metric and LL-based metric
    compute_correlation_LL_BLEU()
