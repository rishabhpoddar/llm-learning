import re
from tokenzer_interface import Tokenizer
from typing import List


class BPETokenizer(Tokenizer):
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = get_tokens_from_text(text)
        ids = [
            self.str_to_int.get(item, self.str_to_int["<|unk|>"])
            for item in preprocessed
        ]
        return ids

    def decode(self, ids):
        text = "".join([self.int_to_str[id] for id in ids])
        return text


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
    vocab.extend(["<|unk|>"])
    return combine_common_tokens(preprocessed, vocab, vocab_size)


def combine_common_tokens(
    preprocessed: List[str], current_vocab: List[str], vocab_size
) -> List[str]:
    if len(current_vocab) >= vocab_size:
        return current_vocab
    new_addition: None | str = None
    new_addition_count = 0
    for current_vocab_left in current_vocab:
        for current_vocab_right in current_vocab:
            pair = current_vocab_left + current_vocab_right
            if pair in current_vocab:
                continue
            count = 0
            for token in preprocessed:
                pass
                if pair in token:  # TODO: This makes the loop really, really slow.
                    count += 1
            if count > new_addition_count:
                new_addition = pair
                new_addition_count = count
    if new_addition:
        current_vocab.append(new_addition)
        return combine_common_tokens(preprocessed, current_vocab, vocab_size)
    return current_vocab


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
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed
