from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .common import GPT2HyperParams, GPT2Params


@dataclass
class GPT2Output:
    # Prediction scores of the language modeling head (scores for each
    # vocabulary token before SoftMax).
    #
    # In language models, "logits" refer to the raw scores obtained from the
    # model's output layer. These scores correspond to each class or token and
    # are not yet in a form that can be interpreted as probabilities. Typically,
    # applying the softmax function to these logits yields the probabilities of
    # each class or token being.
    #
    # Tensor shape: (batch_size, sequence_length, config.vocab_size)
    logits: mx.array

    # Language modeling loss for next-token prediction.
    #
    # In language models, "loss" refers to the computed value that quantifies
    # how well the model's predictions align with the actual target values.
    # It is a measure of the model's performance during training, and it is
    # used to update the model's parameters through backpropagation. The loss
    # is typically computed using a loss function, such as cross-entropy loss,
    #
    # Tensor shape: (1, )
    loss: mx.array | None = None


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
        *,
        compute_loss: bool = False,
    ) -> GPT2Output:
        logits = gpt2(inputs, **self.params, n_head=self.hparams["n_head"])

        if compute_loss and len(inputs) > 1:  # Need at least 2 tokens to compute loss
            # For causal language modeling, targets are the input tokens shifted one position to the right
            # We want to predict the next token for each position in the sequence
            targets = mx.array(inputs[1:])  # Remove the +[0] padding

            # Take logits for all positions except the last one
            # Shape: [1, sequence_length-1, vocab_size]
            logits_for_loss = logits[0, :-1, :]

            # Reshape logits to [sequence_length-1, vocab_size]
            logits_2d = mx.reshape(logits_for_loss, (-1, logits_for_loss.shape[-1]))

            # Use sparse categorical cross-entropy with targets as class indices
            # Targets shape: [sequence_length-1]
            # Logits shape: [sequence_length-1, vocab_size]
            loss = nn.losses.cross_entropy(logits_2d, targets)

            # Average loss across sequence positions
            loss = mx.mean(loss)

            return GPT2Output(logits=logits, loss=loss)

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
