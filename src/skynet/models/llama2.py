import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    """
    Model hyperparameters.
    """
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # Number of heads of the queries;
    n_kv_heads: Optional[int] = None  # Number of heads for the K and V;
    vocab_size: int = -1  # Will be set when we load the tokenizer;
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache:
    max_batch_size: int = 32
    max_seq_len: int = 2048

    # device for calculus:
    device: str = None


def precompute_theta_pos_frequencies(
        head_dim,
        seq_len,
        device,
        theta = 10000.0
):
    """
    Returns precomputed theta position frequencies.

    :param head_dim:
    :param seq_len: The max sequence length.
    :param device:
    :param theta:

    :type head_dim: int
    :type seq_len: int
    :type device: str
    :type theta: float

    :return: a tensor with dim -> (seq_len, head_dim / 2).
    :rtype: torch.Tensor
    """
    # As written in the paper, the dimension of the embedding must be even.
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # Build the theta parameters according
    # to the formula: theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta = 1.0 / (theta ** (theta_numerator / head_dim))
    theta = theta.to(device)

    # Construct the positions (the `m` parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using the outer product.
    # Shape: (seq_len) outer_product*(head_dim / 2) -> (seq_len, head_dim / 2).
    freq = torch.outer(m, theta)
    freq = freq.float()

    # We can compute numbers in the polar form
    # c = R * exp(i * m * theta), where R = 1 as follows:
    # Shape: (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    ones_tensor = torch.ones_like(freq)
    freq_complex = torch.polar(ones_tensor, freq)
    return freq_complex


def apply_rotary_embeddings(x, freq_complex, device):
    """
    Function to apply rotary embeddings.

    :param x:
    :param freq_complex:
    :param device:

    :type x: torch.Tensor
    :type freq_complex: torch.Tensor
    :type device: str

    :return: a tensor with dim -> (B, seq_len, H, head_dim)
    :rtype: torch.Tensor
    """
    # (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim / 2)
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_reshaped = x_reshaped.contiguous()
    x_complex = torch.view_as_complex(x_reshaped)

    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freq_complex = freq_complex.unsqueeze(0)
    freq_complex = freq_complex.unsqueeze(2)

    # (B, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2)
    # = (B, seq_len, H, head_dim / 2)
    x_rotared = x_complex * freq_complex

    # (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotared)
    x_out = x_out.reshape(*x.shape)  # flatten to original tensor (x);
    x_out = x_out.contiguous()  # make tensor contiguous;
    x_out = x_out.type_as(x)  # same type with the original tensor (x);
    x_out = x_out.to(device)  # move to device.
    return x_out


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    :arg dim:
    :arg eps:

    :type dim: `int`
    :type eps: `float`
    """
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # The gamma parameter:
        self.weight = nn.Parameter(torch.one(dim))

    def _normalize(self, x):
        """
        :param x:
        :type x: `torch.Tensor`
        :rtype `torch.Tensor`
        """
        # (B, seq_len, dim) * (B, seq_len, 1) -> (B, seq_len, dim)
        # rsqrt = 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        :param x:
        :type `torch.Tensor`
        :rtype: `torch.Tensor`
        """
        x = self._normalize(x.float()).type_as(x)
        x = self.weight * x
        return x


class EncoderBlock(nn.Module):
    """
    :arg args: The model Hyperparameters.
    :type args: ModelArgs
    """
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the self attention:
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block:
        self.ffm_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freq_complex):
        """
        :param x:
        :param start_pos:
        :param freq_complex:

        :type x: `torch.Tensor`
        :type start_pos: `int`
        :type freq_complex: `torch.Tensor`
        :rtype: `torch.Tensor`
        """
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        attn_norm = self.attention_norm(x)
        h = x + self.attention.forward(attn_norm, start_pos, freq_complex)

        h_norm = self.ffm_norm(h)
        out = h + self.feed_forward(h_norm)
        return out


class Transformer(nn.Module):
    """Definition of the transformer model

    :arg args: The model Hyperparameters.
    :type args: ModelArgs
    """
    def __init__(self, args):
        super().__init__()
        assert args.vocab_size != -1, "Vocab must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Building of Embedding layers:
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        # Building of encoder layers:
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        # Building of output layers:
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_pos_frequencies(
            args.dim // args.n_heads,
            args.max_seq_len*2,
            device=args.device
        )

    def forward(self, tokens, start_pos):
        """
        Function of transformer computation.

        :param tokens:
        :param start_pos:

        :type tokens: torch.Tensor
        :type start_pos: int

        :return:
        :rtype: torch.Tensor
        """
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one (1) token at a time can be processed."
        h = self.tok_embeddings(tokens)  # B x seq_len x dim

        # Retrieve the pairs (m, theta) corresponding to the positions
        # located at start_pos:(start_pos + seq_len).
        freq_complex = self.freqs_complex[start_pos:(start_pos + seq_len)]

        # Consecutively apply all the encoder layers:
        for layer in self.layers:
            h = layer(h, start_pos, freq_complex)
        h = self.norm(h)
        h = self.output(h)

        output = h.float()
        return output
