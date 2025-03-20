from dataclasses import dataclass

import mlx.core as mx

from .common import GPT2HyperParams, GPT2Params


@dataclass
class GPT2Output:
    # Prediction scores of the language modeling head (scores for each vocabulary token before
    # SoftMax).
    #
    # Tensor shape: (batch_size, sequence_length, config.vocab_size)
    # where batch_size is always 1
    logits: mx.array


class GPT2Model:
    params: GPT2Params
    hparams: GPT2HyperParams

    def __init__(self, params: GPT2Params, hparams: GPT2HyperParams):
        self.params = params
        self.hparams = hparams

    def __call__(
        self,
        /,
        inputs: list[int],
    ) -> GPT2Output:
        logits = gpt2(inputs, **self.params, n_head=self.hparams["n_head"])
        return GPT2Output(logits=logits)


def gelu(x):
    return 0.5 * x * (1 + mx.tanh(mx.sqrt(2 / mx.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = mx.exp(x - mx.max(x, axis=-1, keepdims=True))
    return exp_x / mx.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.var(x, axis=-1, keepdims=True)
    x = (x - mean) / mx.sqrt(
        variance + eps
    )  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(
    q, k, v, mask
):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / mx.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = mx.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(
        map(lambda x: mx.split(x, n_head, axis=-1), qkv)
    )  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - mx.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [
        attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)
    ]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = mx.concatenate(
        out_heads, axis=-1
    )  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(
    x, mlp, attn, ln_1, ln_2, n_head
):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(
        layer_norm(x, **ln_1), **attn, n_head=n_head
    )  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(
    inputs, wte, wpe, blocks, ln_f, n_head
):  # [n_seq] -> [batch_size, n_seq, n_vocab]
    # token + positional embeddings
    x = (
        wte[mx.array(inputs)] + wpe[mx.array(list(range(len(inputs))))]
    )  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(
            x, **block, n_head=n_head
        )  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    logits = x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

    # reshape to (batch_size=1, sequence_length, vocab_size)
    return mx.expand_dims(logits, axis=0)  # [n_seq, n_vocab] -> [1, n_seq, n_vocab]
