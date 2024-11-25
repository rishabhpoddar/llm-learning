import torch.nn as nn
import torch


class SelfAttention_V1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        atten_scores = queries @ keys.T  # this is a number
        atten_weights = torch.softmax(
            atten_scores / keys.shape[-1] ** 0.5, dim=-1
        )  # we divide by dimension square root cause it prevents very high number when taking softmax
        context_vector = atten_weights @ values
        return context_vector
