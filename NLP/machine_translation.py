#! usr/bin/env python3
import os
import pickle
import re
import string
import unicodedata
from collections import Counter
import numpy as np

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import trax

# PREPROCESSING


def open_file(file_name: str) -> str:
    with open(file_name, "rt", encoding="utf-8") as f:
        return f.read()


def split_sentences(text: str) -> list[str]:
    return text.strip().split("\n")


def sentence_lengths(sentences: list[str]) -> tuple[int, int]:
    lengths = [len(sentence.split()) for sentence in sentences]
    return min(lengths), max(lengths)


# clean lines
def clean_lines(lines: list[str]) -> list[str]:
    cleaned = []
    # prepare regex for char filtering
    re_print = re.compile("[^%s]" % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans("", "", string.punctuation)

    for line in lines:
        # normalize unicode characters
        line = unicodedata.normalize("NFD", line).encode("ascii", "ignore")
        line = line.decode("UTF-8")
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [word.translate(table) for word in line]
        # remove non-printable chars form each token
        line = [re_print.sub("", w) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(" ".join(line))
    return cleaned


# Excerpt From
# Transformers for Natural Language Processing
# Denis Rothman
# This material may be protected by copyright.


def save_file(file_name: str, data: list[str]) -> None:
    full_path = os.path.join("cleaned", file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as f:
        pickle.dump(data, f)


def load_clean_sentences(file_name: str) -> list[str]:
    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_clean_sentences(sentences: list[str], filename: str) -> None:
    """
    save list of sentences to a pickle file
    """
    full_path = os.path.join("trimmed", filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as f:
        pickle.dump(sentences, f)
        print(f"{len(sentences)} clean sentences saved to {filename}")


def count_words(sentences: list[str]) -> Counter:
    """
    count the number of occurrences of each word in a list of sentences
    """
    vocab = Counter()
    for line in sentences:
        tokens = line.split()
        vocab.update(tokens)
    return vocab


def trim_words(vocab: Counter, min_count: int) -> set:
    """
    remove words that occur less than min_count times
    """
    tokens = [word for word, count in vocab.items() if count >= min_count]
    return set(tokens)


# handling out of vocabulary words
def handle_oov(lines: list[str], vocab: Counter) -> list[str]:
    """
    replace oov words by unknown token
    """
    unknown = "unk"
    new_lines = []
    for line in lines:
        new_tokens = []
        for token in line.split():
            if token in vocab:
                new_tokens.append(token)
            else:
                new_tokens.append(unknown)
        new_line = " ".join(new_tokens)
        new_lines.append(new_line)
    return new_lines


# EVALUATION WITH BLEU

def bleu_eval(references: list[str], candidate: list[str]) -> tuple[float,float]:
    """
    calculate the bleu score between a candidate translation and a list of references
    """
    chencherry = SmoothingFunction()
    smoothed = sentence_bleu(references, candidate, smoothing_function=chencherry.method1)
    bleu_score = sentence_bleu(references, candidate)
    print(f'BLEU score: {bleu_score} \n smoothed: {smoothed}')
    return (
        smoothed,
        bleu_score
    )

# SETTING UP TRANSLATION TRANSFORMER FROM TRAX

def setup_transformer():
    # create transformer model
    model = trax.models.Transformer(
        input_vocab_size=33300, #trimmed vocab size
        d_model=512,
        d_ff=2048,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        max_len=2048,
        mode='predict'
    )
    #initialise pre-trained weights for english french translation
    model.init_from_file('gs://trax-ml/models/translation/ende_wmt32k.pkl.gz', weights_only=True)

    return model

def tokenize(text):
    if isinstance(text, list):
        # Process one sentence at a time
        tokenized_sequences = []
        for sentence in text:
            tokenized = next(trax.data.tokenize(
                iter([sentence]),
                vocab_dir='gs://trax-ml/vocabs/',
                vocab_file='ende_32k.subword'
            ))[0]
            tokenized_sequences.append(tokenized)
        # Find max length for padding
        max_len = max(len(seq) for seq in tokenized_sequences)
        # Pad sequences to same length
        padded_sequences = [np.pad(seq, (0, max_len - len(seq))) for seq in tokenized_sequences]
        return np.array(padded_sequences)
    else:
        # Handle single string input
        tokenized = next(trax.data.tokenize(
            iter([text]),
            vocab_dir='gs://trax-ml/vocabs/',
            vocab_file='ende_32k.subword'
        ))[0]
        return np.array(tokenized)

def decode(model, tokenized):
    tokenized = tokenized[None,:]
    tokenized_translation = trax.supervised.decoding.autoregressive_sample(
        model,
        tokenized,
        temperature=0.0 # controls diversity of results
    )

    tokenized_translation = tokenized_translation[0][:-1]
    translation = trax.data.detokenize(
        tokenized_translation,
        vocab_dir = 'gs://trax-ml/vocabs/',
        vocab_file='ende_32k.subword'
    )
    return translation

# MAIN
def main():
    # preprocessing
    files = [
        # "./NLP/French English/europarl-v7.fr-en.en",
        # "./NLP/French English/europarl-v7.fr-en.fr",
        "DE EN/europarl-v7.de-en.de",
        "DE EN/europarl-v7.de-en.en",
    ]
    for file in files:
        data = open_file(file)
        data = split_sentences(data)
        min_len, max_len = sentence_lengths(data)
        print(
            f"{file}: min len:{min_len} - max len:{max_len}, total num sentences:{len(data)}"
        )
        data = clean_lines(data)
        save_file(file + ".clean", data)

    cleaned_files = ["cleaned/" + file + ".clean" for file in files]
    for file in cleaned_files:
        lines = load_clean_sentences(file)
        vocab = count_words(lines)
        print(f"{file}: {len(vocab)}")
        vocab = trim_words(vocab, min_count=5)
        print(f"Trimmed {file}: {len(vocab)}")
        lines = handle_oov(lines, vocab)
        save_clean_sentences(lines, file + ".trimmed")
        print("CHECK:")
        for i in range(20):
            print("line", i, ":", lines[i])  # print the first 20 lines to check

    # TRANSLATION
    model = setup_transformer()
    for file in cleaned_files:
        lines = load_clean_sentences('trimmed/' + file + ".trimmed")
        tokenized = tokenize(lines)
        translation = decode(model, tokenized)
        save_file("translated/" + file + ".translated", translation)

    # CHECK
    translated_files = ["translated/" + file + ".translated" for file in files]
    for file in translated_files:
        lines = load_clean_sentences(file)
        print(f"{file}: {len(lines)}")
        for i in range(20):
            print("line", i, ":", lines[i])  # print the first 20 lines to check


if __name__ == "__main__":
    main()
