import torch

# This is from chapter 3.3.1 in the book.


def transform_input(inputs):
    # compute attention scores
    atten_scores = inputs @ inputs.T

    # normalise attention scores using softmax
    atten_scores = torch.softmax(atten_scores, dim=-1)

    # compute context vector
    context_vector = atten_scores @ inputs

    return context_vector
