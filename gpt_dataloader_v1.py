import torch
from torch.utils.data import Dataset, DataLoader
from byte_pair_encoding_tokenizer import get_tokenizer
from token_embedding import TokenEmbedding


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        self.token_embedding = TokenEmbedding(tokenizer.vocab_size, 256, max_length)

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.token_embedding.get_embeddings(
            self.input_ids[idx]
        ), self.token_embedding.get_embeddings(self.target_ids[idx])


def create_data_loader(
    txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True
):
    tokenizer = get_tokenizer(txt)
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size, shuffle, drop_last=drop_last)
