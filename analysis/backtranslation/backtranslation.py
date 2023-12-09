from transformers import MarianTokenizer, MarianMTModel
import tqdm
import random
import argparse


def save(snt, prd, trglang):
    """
    Append backtranslations to existing file.
    Args:
        - snt (list of str)
        - prd (list of str)
        - trglang (str): nl | de | fr | it | es
    """
    with open(
        f"backtranslation/translations_parallel_opus.en-{trglang}",
        "a",
        encoding="utf-8",
    ) as f:
        for s, p in zip(snt, prd):
            f.write(f"{s.strip()}\t{p.strip()}\n")


def translate(sentences, trglang):
    """
    Translate a list of sentences in <trglang> back to English.
    Args:
        - sentences (list of str)
        - trlang (str): nl | de | fr | it | es
    """
    mname = f"Helsinki-NLP/opus-mt-{trglang}-en"
    model = MarianMTModel.from_pretrained(mname)
    model.cuda()
    tok = MarianTokenizer.from_pretrained(mname)
    sentences = list(set(sentences))
    batches = [
        [t.replace("@@ ", "") for t in sentences[i : i + 16]]
        for i in range(0, len(sentences), 16)
    ]

    predictions = []
    for k, batch_sentences in enumerate(batches):
        if k % 50 == 0:
            print(f"{k} / {len(batches)}")
        batch = tok(
            batch_sentences,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            padding=True,
        )
        for key in batch:
            batch[key] = batch[key].cuda()
        outputs = model.generate(
            **batch, num_beams=1, max_length=512, return_dict_in_generate=True
        )
        predictions = tok.batch_decode(outputs["sequences"], skip_special_tokens=True)
        save(batch_sentences, predictions, trglang)
    return predictions


def main(trglang):
    """
    Obtain backtranslations for EN-<trglang>
    Args:
        - trglang (str): nl | de | fr | it | es
    """
    train = open(
        f"../data/parallel_opus/en-{trglang}/parallel_opus.{trglang}", encoding="utf-8"
    ).readlines()
    dataset = [x.strip() for x in train]
    translate(dataset, trglang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trglang", type=str)
    args = parser.parse_args()
    main(args.trglang)
