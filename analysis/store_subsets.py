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
md = MosesDetokenizer(lang='nl')


def store_subsets_to_file(lang, postfix, num_tokens):
    """
    Store subsets that have 750k tokens removed to file.
    Args:
        - lang (str): de | nl | it | es | fr
        - postfix (str): hyp | ref
        - num_tokens: number of tokens to remove
    """
    seed = 1
    src = open(f"../data/parallel_opus2/en-{lang}/train.en", encoding="utf-8").readlines()
    trg = open(f"../data/parallel_opus2/en-{lang}/train.{lang}", encoding="utf-8").readlines()
    print(len(set(zip(src, trg))))
    counts = Counter(zip([s.strip() for s in src], [t.strip() for t in trg]))

    filename = f"memorisation_pickled/en-{lang}/counterfactual_memorisation_{postfix}.pickle"
    memorisation, train_scores, test_scores = pickle.load(open(filename, 'rb'))
    print(len(memorisation))
    if "hyp" in postfix:
        memorisation = {k: v/100 for k, v in memorisation.items()}
        train_scores = {k: v/100 for k, v in train_scores.items()}
        test_scores = {k: v/100 for k, v in test_scores.items()}

    coordinate_mapping = {}

    k = 0
    for i in range(1, 11):
        for j in range(1, 11):
            if j > i:
                continue
            coordinate = (i/10, j/10)

            print(k, i, j)
            train, excluded = [], []
            random.seed(seed)
            np.random.seed(seed)
            items = list(memorisation.items())
            random.shuffle(items)
            sentences, scores = zip(*items)

            # Use KDTree for fast nearest neighbour lookup
            n = 75000
            scores = [(train_scores[s], test_scores[s]) for s in sentences]
            kdtree = KDTree(scores)
            # Retrieve the closest 75k examples
            distances, indices = kdtree.query(coordinate, k=n)
            source_tokens = 0
            all_indices = list(indices) + \
                [sent_idx for sent_idx in range(len(sentences))
                if sent_idx not in indices]
            # Now EXCLUDE sentences until we have the right number of tokens
            for index in all_indices:
                if index in indices and source_tokens <= num_tokens:
                    for _ in range(counts[sentences[index]]):
                        excluded.append((index, scores[index]))
                        source_tokens += len(sentences[index][0].split())
                else:
                    # Collect the remaining examples in a new training set
                    for _ in range(counts[sentences[index]]):
                        train.append(sentences[index])

            # Collect the ids of the examples excluded in coordinate mapping
            coordinate_mapping[(k, i, j, coordinate)] = excluded
            random.shuffle(train)

            # Report how many examples were removed
            tokens = [w for l in excluded for w in sentences[l[0]][0].split()]
            print(len(train), len(excluded), len(tokens))
            with open(f"subsets/en-{lang}/subset={k}_{postfix}.en", 'w', encoding="utf-8") as f:
                for l, _ in train:
                    f.write(l + "\n")
            with open(f"subsets/en-{lang}/subset={k}_{postfix}.{lang}", 'w', encoding="utf-8") as f:
                for _, l in train:
                    f.write(l + "\n")
            pickle.dump(
                coordinate_mapping,
                open(f"subsets/en-{lang}/coordinate_mappings_{postfix}.pickle", 'wb'))
            k += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--postfix", type=str, required=True)
    parser.add_argument("--num_tokens", type=int, default=750000)
    args = parser.parse_args()
    store_subsets_to_file(args.lang, args.postfix, args.num_tokens)
