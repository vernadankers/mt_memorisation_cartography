import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from shapely.geometry import Point, Polygon
from map_features import Example


def load(examples, with_targets=False):
    """
    Given data from the customised Example class, load X (datapoints' features)
    and y (memorisation metrics) as vectors.
    Args:
        - examples (list of Example objects)
        - with_targets (bool): whether these examples have targets
    Returns:
        - X: array of numerical features
        - y: array of floats (memorisation metrics)
    """
    X, y = [], []
    for e in examples:
        if None in list(e.numerical_features.values()):
            continue
        sample = []
        for f in e.numerical_features:
            sample.append(e.numerical_features[f])
        X.append(sample)
        targets = []
        if with_targets:
            for metric in ["train_likelihood_ref", "test_likelihood_ref",
                           "counterfactual_memorisation_ref"]:
                targets.append(getattr(e, metric))
            y.append(targets)
    return X, y


def train_model():
    """Train a model on En-Nl memorisation data.

    Returns:
        MLPRegressor
    """
    examples_nl = pickle.load(open(
        "memorisation_pickled/en-nl/examples_nl.pickle", 'rb')).values()
    model = MLPRegressor(
        hidden_layer_sizes=(100, 100), max_iter=20, verbose=False)
    X, y = load(examples_nl, with_targets=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1, test_size=0.8)
    model.fit(X_train, y_train)
    print(f"R^2 train: {model.score(X_train, y_train)}, "+\
          f"test: {model.score(X_test, y_test)}")
    return model



def predict_on_30M(model):
    examples_xl = list(pickle.load(open(
        "memorisation_pickled/en-nl/examples_nl.pickle", 'rb')
    ).values())

    corpus_scored = dict()
    for i in tqdm.tqdm_notebook(range(0, 900000, 1000)):
        examples = examples_xl[i:i+1000]
        X, _ = load(examples, with_targets=False)
        print(X)
        prds = model.predict(X)
        [train, test] = list(zip(*prds))[:2]
        for e, x_, y_ in zip(examples, train, test):
            corpus_scored[e.source_tokenised, e.target_tokenised] = (x_, y_)
    return corpus_scored



def get_specialised_corpus(corpus_scored, num_tokens_rand, coords, name):
    poly = Polygon(coords)
    x, y = [], []
    specialised_corpus = []
    for e, c in tqdm.tqdm_notebook(corpus_scored.items()):
        if poly.contains(Point(c[0], c[1])):
            x.append(c[0])
            y.append(c[1])
            specialised_corpus.append(e)
    sns.jointplot(x=x, y=y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f"corpus_{name}.pdf", bbox_inches="tight")

    random.shuffle(specialised_corpus)
    with open(f"specialised_corpora/{name}.en", 'w', encoding="utf-8") as fs, \
         open(f"specialised_corpora/{name}.nl", 'w', encoding="utf-8") as ft:
            num_tokens_spec = 0
            for s, t in specialised_corpus:
                num_tokens_spec += len(s.split())
                fs.write(s + '\n')
                ft.write(t + '\n')
                if num_tokens_spec > num_tokens_rand:
                    break
    print(f"Tokens in new corpus: {num_tokens2}, "+
          f"tokens in random: {num_tokens_rand}")




if __name__ == "__main__":
    model = train_model()
    corpus_scored = predict_on_30M(model)

    # Get a random corpus of 1M examples
    with open("specialised_corpora/random.en", 'w', encoding="utf-8") as fs, \
         open("specialised_corpora/random.nl", 'w', encoding="utf-8") as ft:
        for s, t in random.sample(list(corpus_scored.keys()), 1000000):
            fs.write(s + '\n')
            ft.write(t + '\n')
    num_tokens_rand = len([
        w for l in open("specialised_corpora/random.en", encoding='utf-8').readlines()
        for w in l.split()])


    get_specialised_corpus(
        corpus_scored, num_tokens_rand,
        coords=[(0.6, 0.0), (1, 1), (1, 0)],
        name="bleu_hal")


    get_specialised_corpus(
        corpus_scored, num_tokens_rand,
        coords = [(0, 0), (0.4, 0.4), (1, 0.4), (1, 0)],
        name="log_probability")
