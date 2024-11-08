import re
from tokenzer_interface import Tokenizer


class SimpleTokenizerV1(Tokenizer):
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = get_tokens_from_text(text)
        ids = [self.str_to_int[item] for item in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])
        # this makes text like "This , is awesome" to "This, is awesome"
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


def get_tokenizer(text):
    # first we convert "This is an example" to ["This", "is", "an", "example"]
    preprocessed = get_tokens_from_text(text)
    # then we get the unique tokens
    vocab = sorted(list(set(preprocessed)))
    # then we convert them to ids
    vocab_to_int = {token: i for i, token in enumerate(vocab)}
    return SimpleTokenizerV1(vocab_to_int)


def get_tokens_from_text(text):
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed
