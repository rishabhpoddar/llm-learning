import re
from tokenzer_interface import Tokenizer
from typing import List

# Algorithm: https://huggingface.co/learn/nlp-course/en/chapter6/5


class BPETokenizer(Tokenizer):
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.sorted_vocab_list = list(vocab.keys())
        self.sorted_vocab_list.sort(
            key=lambda x: -1 * len(x)
        )  # -1 cause we want longest first
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        if text == "":
            return []
        preprocessed = get_tokens_from_text(text)
        ids = []
        for token in preprocessed:
            if token == "":
                continue
            found = False
            for vocab_token in self.sorted_vocab_list:
                if vocab_token in token:
                    ids.extend(self.encode(token[: token.index(vocab_token)]))
                    ids.append(self.str_to_int[vocab_token])
                    ids.extend(
                        self.encode(
                            token[token.index(vocab_token) + len(vocab_token) :]
                        )
                    )
                    found = True
                    break
            if not found:
                raise Exception(f"Token {token} not found in vocab")
        return ids

    def decode(self, ids):
        return "".join([self.int_to_str[id] for id in ids])


def get_tokenizer(text: str, vocab_size=100):
    # first we convert "This is an example" to ["This", "is", "an", "example"]
    preprocessed = get_tokens_from_text(text)
    # then we get the unique tokens
    vocab = create_vocab(preprocessed, vocab_size)
    # then we convert them to ids
    vocab_to_int = {token: i for i, token in enumerate(vocab)}
    return BPETokenizer(vocab_to_int)


def create_vocab(preprocessed: List[str], vocab_size):
    vocab = create_initial_vocab(preprocessed, vocab_size)
    split = [list(word) for word in preprocessed]
    return combine_common_tokens(split, vocab, vocab_size)


def combine_common_tokens(
    current_split: List[str], current_vocab: List[str], vocab_size
) -> List[str]:
    if len(current_vocab) >= vocab_size:
        return current_vocab
    split_frequency = {}
    for split in current_split:
        for i in range(len(split) - 1):
            pair = split[i] + split[i + 1]
            split_frequency[pair] = split_frequency.get(pair, 0) + 1
    most_frequent_pair = max(split_frequency, key=split_frequency.get)
    current_vocab.append(most_frequent_pair)
    for i, split in enumerate(current_split):
        for j in range(len(split) - 1):
            if split[j] + split[j + 1] == most_frequent_pair:
                current_split[i] = (
                    current_split[i][:j]
                    + [most_frequent_pair]
                    + current_split[i][j + 2 :]
                )
    return combine_common_tokens(current_split, current_vocab, vocab_size)


def create_initial_vocab(preprocessed: List[str], vocab_size=5000):
    vocab = []
    for token in preprocessed:
        for char in token:
            if char not in vocab:
                vocab.append(char)
            if len(vocab) >= vocab_size:  # >= cause we also want to add <|unk|>
                raise Exception(
                    "Increase vocab size to at least accommodate the base vocab"
                )
    return vocab


def get_tokens_from_text(text: str) -> List[str]:
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
    return preprocessed
