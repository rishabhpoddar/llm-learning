import torch


class TokenEmbedding:
    def __init__(self, vocab_size, embedding_dim, context_length):
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        self.position_embedding_layer = torch.nn.Embedding(
            context_length, embedding_dim
        )

        # this steps converts the above into a tensor that can be added to the token embeddings
        self.position_embeddings = self.position_embedding_layer(
            torch.arange(context_length)  # tensor from 0..context_length - 1
        )

    def get_embeddings(self, input_tokens):
        token_embeddings = self.embedding(
            input_tokens
        )  # this does a lookup. Token ID X picks the row X in the embedding matrix
        return token_embeddings + self.position_embeddings
