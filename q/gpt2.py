from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .common import GPT2HyperParams, GPT2LayerNormParams, GPT2Params


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

    # List of (past_k, past_v) cache for each block
    past_key_values: list[tuple[mx.array, mx.array]]

    # Language modeling loss for next-token prediction.
    #
    # In language models, "loss" refers to the computed value that quantifies
    # how well the model's predictions align with the actual target values.
    # It is a measure of the model's performance during training, and it is
    # used to update the model's parameters through back propagation. The loss
    # is typically computed using a loss function, such as cross-entropy loss,
    #
    # Tensor shape: ()
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
        # inputs: (batch_size, sequence_length)
        inputs: mx.array,
        *,
        compute_loss: bool = False,
        past_key_values: Optional[list[tuple[mx.array, mx.array]]] = None,
    ) -> GPT2Output:
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must be 2D (batch_size, sequence_length) tensor, but got {inputs.ndim} ndim."
            )

        logits, kv = gpt2(
            inputs,
            **self.params,
            n_head=self.hparams["n_head"],
            past_key_values=past_key_values,
        )

        if (
            compute_loss and inputs.shape[1] > 1
        ):  # Need at least 2 tokens to compute loss
            # For causal language modeling, targets are the input tokens shifted one position to the right
            # We want to predict the next token for each position in the sequence
            targets = inputs[:, 1:]  # (batch_size, sequence_length-1)

            # Take logits for all positions except the last one
            # Shape: (batch_size, sequence_length-1, vocab_size)
            logits_for_loss = logits[:, :-1, :]

            # Reshape logits to (sequence_length-1, vocab_size)
            logits_2d = mx.reshape(logits_for_loss, (-1, logits_for_loss.shape[-1]))
            targets_1d = mx.reshape(targets, (-1,))

            # Use sparse categorical cross-entropy with targets as class indices
            # Targets shape: (sequence_length-1)
            # Logits shape: (sequence_length-1, vocab_size)
            loss = nn.losses.cross_entropy(logits_2d, targets_1d)

            # Average loss across sequence positions
            loss = mx.mean(loss)

            return GPT2Output(logits=logits, loss=loss, past_key_values=kv)

        return GPT2Output(logits=logits, past_key_values=kv)


def gelu(x):
    return 0.5 * x * (1 + mx.tanh(mx.sqrt(mx.array(2 / mx.pi)) * (x + 0.044715 * x**3)))


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


def linear(
    x: mx.array, w: mx.array, b: mx.array
) -> mx.array:  # (m, in), (in, out), (out) -> (m, out)
    return x @ w + b


def ffn(x, c_fc, c_proj):  # (n_seq, n_embd) -> (n_seq, n_embd)
    # project up
    a = gelu(linear(x, **c_fc))  # (n_seq, n_embd) -> (n_seq, 4*n_embd)

    # project back down
    x = linear(a, **c_proj)  # (n_seq, 4*n_embd) -> (n_seq, n_embd)

    return x


def attention(q, k, v, mask):
    # q, k, v: (batch_size, seq_len, head_dim)
    # mask: (1, seq_len, seq_len) → broadcast 可能
    scores = mx.matmul(q, k.transpose(0, 2, 1)) / mx.sqrt(q.shape[-1])
    scores = scores + mask
    weights = softmax(scores)  # (batch, seq_len, seq_len)
    return mx.matmul(weights, v)  # (batch, seq_len, head_dim)


def mha(
    x: mx.array, c_attn: dict, c_proj: dict, *, n_head: int, past_k=None, past_v=None
) -> tuple[
    mx.array,  # output: (batch, new_seq_len, emb)
    mx.array,  # k: (batch, total_seq_len, n_head, head_dim)
    mx.array,  # v: (batch, total_seq_len, n_head, head_dim)
]:
    # qkv projection: (batch, seq_len, 3*emb)
    qkv = linear(x, **c_attn)
    q, k, v = mx.split(qkv, 3, axis=-1)

    batch, seq_len, embed_dim = q.shape
    head_dim = embed_dim // n_head

    # Reshape q, k, v to (batch, seq_len, n_head, head_dim)
    q = mx.reshape(q, (batch, seq_len, n_head, head_dim))
    k = mx.reshape(k, (batch, seq_len, n_head, head_dim))
    v = mx.reshape(v, (batch, seq_len, n_head, head_dim))

    # If past keys/values exist, concatenate them along the sequence dimension (axis=1)
    if past_k is not None and past_v is not None:
        # Expected shape for past_k, past_v: (batch, past_seq_len, n_head, head_dim)
        k = mx.concatenate([past_k, k], axis=1)
        v = mx.concatenate([past_v, v], axis=1)

    total_seq_len = k.shape[1]  # total sequence length including past
    new_seq_len = q.shape[1]  # sequence length for the new input (usually 1)

    # Rearrange tensors for attention: (batch, n_head, seq_len, head_dim)
    # We only transpose for the attention computation.
    q_t = mx.transpose(q, (0, 2, 1, 3))
    k_t = mx.transpose(k, (0, 2, 1, 3))
    v_t = mx.transpose(v, (0, 2, 1, 3))

    # Scaled dot-product attention
    scores = mx.matmul(q_t, mx.transpose(k_t, (0, 1, 3, 2))) / mx.sqrt(
        mx.array(head_dim)
    )

    # Create causal mask with shape (1, 1, new_seq_len, total_seq_len)
    causal_mask = (1 - mx.tri(new_seq_len, total_seq_len, k=0, dtype=x.dtype)) * -1e10
    causal_mask = causal_mask.reshape((1, 1, new_seq_len, total_seq_len))
    scores = scores + causal_mask

    # Compute attention weights and context
    weights = softmax(scores)  # (batch, n_head, new_seq_len, total_seq_len)
    context = mx.matmul(weights, v_t)  # (batch, n_head, new_seq_len, head_dim)

    # Rearrange context back to (batch, new_seq_len, n_head, head_dim)
    context = mx.transpose(context, (0, 2, 1, 3))
    # Reshape to (batch, new_seq_len, embed_dim)
    context = mx.reshape(context, (batch, new_seq_len, embed_dim))
    output = linear(context, **c_proj)

    # Return output and the keys/values in original shape:
    # k and v: (batch, total_seq_len, n_head, head_dim)
    return output, k, v


def transformer_block(
    x, mlp, attn, ln_1, ln_2, *, n_head: int, past_k=None, past_v=None
) -> tuple[
    mx.array,  # output: (batch, seq_len, emb)
    mx.array,  # k: (batch, total_seq_len, n_head, head_dim)
    mx.array,  # v: (batch, total_seq_len, n_head, head_dim)
]:
    # multi-head causal self attention
    attn_output, k, v = mha(
        layer_norm(x, **ln_1), **attn, n_head=n_head, past_k=past_k, past_v=past_v
    )
    x = x + attn_output

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # (n_seq, n_embd) -> (n_seq, n_embd)

    return x, k, v


def gpt2(
    inputs: mx.array,
    wte: mx.array,
    wpe: mx.array,
    blocks: list,
    ln_f: GPT2LayerNormParams,
    *,
    n_head: int,
    # List of (past_k, past_v) cache for each block
    past_key_values: Optional[list[tuple[mx.array, mx.array]]] = None,
) -> tuple[
    # logits
    mx.array,
    # kv
    list[tuple[mx.array, mx.array]],
]:
    seq_length: int = int(inputs.shape[1])

    # token + positional embeddings
    token_embed = wte[inputs]  # (batch_size, sequence_length, n_embd)
    pos_indices = mx.array(list(range(seq_length)))
    pos_embed = wpe[pos_indices]  # (sequence_length, n_embd)
    pos_embed = mx.expand_dims(pos_embed, axis=0)  # (1, sequence_length, n_embd)

    x = token_embed + pos_embed  # (batch_size, sequence_length, n_embd)
    new_past_kv: list[tuple[mx.array, mx.array]] = []

    # forward pass through n_layer transformer blocks
    for i, block in enumerate(blocks):
        past_k, past_v = (
            past_key_values[i] if past_key_values is not None else (None, None)
        )
        x, k, v = transformer_block(
            x, **block, n_head=n_head, past_k=past_k, past_v=past_v
        )
        new_past_kv.append((k, v))

    # projection to vocab
    x = layer_norm(x, **ln_f)  # (n_seq, n_embd) -> (n_seq, n_embd)
    logits = x @ wte.T  # (n_seq, n_embd) -> (n_seq, n_vocab)

    # reshape to (batch_size=1, sequence_length, vocab_size)
    return logits, new_past_kv
