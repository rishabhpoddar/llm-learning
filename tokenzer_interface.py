from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, ids):
        pass
