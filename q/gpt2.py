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
    loss: Optional[mx.array] = None


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

        if compute_loss and inputs.shape[1] > 1:  # 損失計算には最低2トークン必要
            targets = inputs[:, 1:]  # 右シフトしたターゲット
            logits_for_loss = logits[:, :-1, :]  # 最後のトークンは除く
            logits_2d = mx.reshape(logits_for_loss, (-1, logits_for_loss.shape[-1]))
            targets_1d = mx.reshape(targets, (-1,))

            # Use sparse categorical cross-entropy with targets as class indices
            # Targets shape: (sequence_length-1)
            # Logits shape: (sequence_length-1, vocab_size)
            # then, average loss across sequence positions
            loss = nn.losses.cross_entropy(logits_2d, targets_1d)
            loss = mx.mean(loss)

            return GPT2Output(logits=logits, loss=loss, past_key_values=kv)

        return GPT2Output(logits=logits, past_key_values=kv)


def gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.tanh(mx.sqrt(mx.array(2 / mx.pi)) * (x + 0.044715 * x**3)))


def softmax(x: mx.array) -> mx.array:
    exp_x = mx.exp(x - mx.max(x, axis=-1, keepdims=True))
    return exp_x / mx.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x: mx.array, g: mx.array, b: mx.array, eps: float = 1e-5) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / mx.sqrt(variance + eps)
    return g * normalized + b


def linear(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    return x @ w + b


def ffn(x: mx.array, c_fc: dict, c_proj: dict) -> mx.array:
    hidden = gelu(linear(x, **c_fc))  # (n_seq, n_embd) -> (n_seq, 4*n_embd)
    return linear(hidden, **c_proj)  # (n_seq, 4*n_embd) -> (n_seq, n_embd)


def mha(
    x: mx.array,
    c_attn: dict,
    c_proj: dict,
    *,
    n_head: int,
    past_k: Optional[mx.array] = None,
    past_v: Optional[mx.array] = None,
) -> tuple[
    mx.array,  # output: (batch, new_seq_len, emb)
    mx.array,  # k: (batch, total_seq_len, n_head, head_dim)
    mx.array,  # v: (batch, total_seq_len, n_head, head_dim)
]:
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

    total_seq_len = k.shape[1]
    past_seq_len = total_seq_len - seq_len

    # Rearrange tensors for attention: (batch, n_head, seq_len, head_dim)
    # We only transpose for the attention computation.
    q_t = mx.transpose(q, (0, 2, 1, 3))
    k_t = mx.transpose(k, (0, 2, 1, 3))
    v_t = mx.transpose(v, (0, 2, 1, 3))

    # Scaled dot-product attention
    scores = mx.matmul(q_t, mx.transpose(k_t, (0, 1, 3, 2))) / mx.sqrt(
        mx.array(head_dim)
    )

    # Create a mask that allows each position to attend to all previously generated tokens
    # but not to future tokens. For KV cache, we need a mask that allows attendance to
    # past positions.
    if past_seq_len > 0:
        mask_future = (1 - mx.tri(seq_len, seq_len, k=0, dtype=x.dtype)) * -1e10
        mask = mx.concatenate(
            [
                mx.zeros((1, 1, seq_len, past_seq_len)),  # No mask for past positions
                mask_future.reshape(
                    (1, 1, seq_len, seq_len)
                ),  # Only mask future positions in current sequence
            ],
            axis=3,
        )
    else:
        # Standard causal mask when there's no past
        mask = (1 - mx.tri(seq_len, total_seq_len, k=0, dtype=x.dtype)) * -1e10
        mask = mask.reshape((1, 1, seq_len, total_seq_len))

    scores += mask

    # Compute attention weights and context
    weights = softmax(scores)
    context = mx.matmul(weights, v_t)
    # Rearrange context back to (batch, seq_len, n_head, head_dim)
    context = mx.transpose(context, (0, 2, 1, 3))
    # Reshape to (batch, seq_len, embed_dim)
    context = mx.reshape(context, (batch, seq_len, embed_dim))
    output = linear(context, **c_proj)

    # Return output and the keys/values in original shape:
    # k and v: (batch, total_seq_len, n_head, head_dim)
    return output, k, v


def transformer_block(
    x: mx.array,
    mlp: dict,
    attn: dict,
    ln_1: dict,
    ln_2: dict,
    *,
    n_head: int,
    past_k: Optional[mx.array] = None,
    past_v: Optional[mx.array] = None,
) -> tuple[
    mx.array,
    mx.array,  # k: (batch, total_seq_len, n_head, head_dim)
    mx.array,  # v: (batch, total_seq_len, n_head, head_dim)
]:
    # Pre-norm attention と residual connection
    norm_x = layer_norm(x, **ln_1)
    attn_output, k, v = mha(norm_x, **attn, n_head=n_head, past_k=past_k, past_v=past_v)
    x = x + attn_output

    # Pre-norm Feed-Forward と residual connection
    norm_x = layer_norm(x, **ln_2)
    x = x + ffn(norm_x, **mlp)
    return x, k, v


def gpt2(
    inputs: mx.array,
    wte: mx.array,
    wpe: mx.array,
    blocks: list[dict],
    ln_f: GPT2LayerNormParams,
    *,
    n_head: int,
    past_key_values: Optional[list[tuple[mx.array, mx.array]]] = None,
) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
    batch_size, seq_length = inputs.shape

    # Set the position indices correctly for KV cache
    pos_start = 0
    if past_key_values is not None and past_key_values[0][0] is not None:
        pos_start = past_key_values[0][0].shape[1]

    token_embed = wte[inputs]

    # Generate position indices using mx.arange for better readability and efficiency
    pos_indices = mx.arange(pos_start, pos_start + seq_length)
    pos_embed = wpe[pos_indices]
    pos_embed = mx.expand_dims(pos_embed, axis=0)
    x = token_embed + pos_embed

    new_past_kv: list[tuple[mx.array, mx.array]] = []
    for i, block in enumerate(blocks):
        past_k, past_v = (
            past_key_values[i] if past_key_values is not None else (None, None)
        )
        x, k, v = transformer_block(
            x, **block, n_head=n_head, past_k=past_k, past_v=past_v
        )
        new_past_kv.append((k, v))

    x = layer_norm(x, **ln_f)
    logits = x @ wte.T
    return logits, new_past_kv
