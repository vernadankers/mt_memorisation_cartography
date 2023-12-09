# mt_memorisation_cartography

Code for "Memorisation Cartography: Mapping out the Memorisation-Generalisation Continuum in Neural Machine Translation", to appear at EMNLP-2023 in Singapore, in December of 2023.

Visit our demo: https://memorisation-mt-demo.github.io/memorisation-mt-demo/demo.html

#### Preparations
- Ensure our parallel corpus is underneath `data/parallel_opus` (contact us for the corpus, `vernadankers@gmail.com`)
- Install packages with `conda env create -f env.yml`
- `cd fairseq` and `pip install -e .`

#### Section 3: Model training to compute memorisation metrics
1. In `modelling` run `bash submit_scripts.sh memorisation-training <language>`, followed by `bash submit_scripts.sh memorisation-testing <language>`.
2. Afterwards, in `analysis` run `python compute_memorisation.py --trglang <language> --postfix ref`. You will now have computed memorisation metrics under `analysis/memorisation_pickled/en-<language>`.

#### Section 4: Data characterisation
0. To prepare for the analysis three types of data are required:
    - the predictions that are the outcome of the memorisation pretraining
    - obtain alignments using `eflomal` (https://github.com/robertostling/eflomal), and insert them in `analysis/alignments/`
    - obtain backtranslations in the `analysis/backtranslation` folder using `python backtranslation.py --trglang <language>`
1. Afterwards, you can "map out" the data by running `python map_features.py --trglang <language>`. You will now have examples and their features stored under `memorisation_pickled/en-<language>/examples_<language>.pickle`.
2. Functionality for the analyses of section 4 are contained in `section4.py`.

#### Section 5: Approximating memorisation measures
1. In `modelling` run `bash submit_scripts.sh training-dynamics-training <language>`, followed by `bash submit_scripts.sh training-dynamics-testing <language>`.
2. Process signals, train MLPs and visualise results for section 5 using `section5.py`.

#### Section 6: Memorisation and performance
1. Create subsets with examples held out, by running `python store_subsets.py` for section 6.1. In `modelling` run `bash submit_scripts.sh performance-impact-training <language>`, followed by `bash submit_scripts.sh performance-impact-testing <language>`.
2. Create "specialised corpora" using `python store_specialised_corpora.py` for section 6.2. In `modelling` run `bash submit_scripts.sh improved-training <language>`, followed by `bash submit_scripts.sh improved-testing <language>`.

##### Reference

```
@inproceedings{dankers2023memorisation,
  title={Memorisation Cartography: Mapping out the Memorisation-Generalisation Continuum in Neural Machine Translation},
  author={Dankers, Verna and Titov, Ivan and Hupkes, Dieuwke},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2023}
}
```
