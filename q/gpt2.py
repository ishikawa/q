from dataclasses import dataclass

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
    ) -> GPT2Output:
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must be 2D (batch_size, sequence_length) tensor, but got {inputs.ndim} ndim."
            )

        logits = gpt2(inputs, **self.params, n_head=self.hparams["n_head"])

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

            return GPT2Output(logits=logits, loss=loss)

        return GPT2Output(logits=logits)


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


# multi-head attention
def mha(
    x: mx.array, c_attn: dict, c_proj: dict, *, n_head: int, past_k=None, past_v=None
) -> tuple[
    mx.array,  # (batch, seq_len, emb)
    mx.array,  # k
    mx.array,  # v
]:
    # qkv projection
    qkv = linear(x, **c_attn)  # (batch, seq_len, 3*emb)
    q, k, v = mx.split(qkv, 3, axis=-1)

    # split into heads → list of length n_head; each (batch, seq_len, head_dim)
    q_heads = mx.split(q, n_head, axis=-1)
    k_heads = mx.split(k, n_head, axis=-1)
    v_heads = mx.split(v, n_head, axis=-1)

    # If `past_k` and `past_v` are provided, concatenate them with the new keys and values
    new_k_heads: mx.array = k_heads
    new_v_heads: mx.array = v_heads

    if past_k is not None and past_v is not None:
        # Concatenate past keys and values with the new keys and values
        ks: list[mx.array] = []
        vs: list[mx.array] = []

        for i in range(n_head):
            new_k = mx.concatenate([past_k[i], k_heads[i]], axis=1)
            new_v = mx.concatenate([past_v[i], v_heads[i]], axis=1)
            ks.append(new_k)
            vs.append(new_v)

        new_k_heads = mx.concatenate(ks)
        new_v_heads = mx.concatenate(vs)

    total_seq_len = new_k_heads[0].shape[1]

    # causal mask: (seq_len, seq_len) → (1, seq_len, seq_len)
    causal_mask = (1 - mx.tri(total_seq_len, total_seq_len, k=0, dtype=x.dtype)) * -1e10
    causal_mask = mx.expand_dims(causal_mask, axis=0)

    # per‑head attention
    out_heads = [
        attention(qh, kh, vh, causal_mask)
        for qh, kh, vh in zip(q_heads, new_k_heads, new_v_heads)
    ]  # list of (batch, seq_len, head_dim)

    # concat heads → (batch, seq_len, emb)
    concat = mx.concatenate(out_heads, axis=-1)
    output = linear(concat, **c_proj)

    return output, new_k_heads, new_v_heads


def transformer_block(
    x, mlp, attn, ln_1, ln_2, n_head
):  # (n_seq, n_embd) -> (n_seq, n_embd)
    # multi-head causal self attention
    m, _k, _v = mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + m

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # (n_seq, n_embd) -> (n_seq, n_embd)

    return x


def gpt2(
    inputs: mx.array,
    wte: mx.array,
    wpe: mx.array,
    blocks: list,
    ln_f: GPT2LayerNormParams,
    n_head: int,
):
    seq_length: int = int(inputs.shape[1])

    # token + positional embeddings
    token_embed = wte[inputs]  # (batch_size, sequence_length, n_embd)
    pos_indices = mx.array(list(range(seq_length)))
    pos_embed = wpe[pos_indices]  # (sequence_length, n_embd)
    pos_embed = mx.expand_dims(pos_embed, axis=0)  # (1, sequence_length, n_embd)

    x = token_embed + pos_embed  # (batch_size, sequence_length, n_embd)

    # forward pass through n_layer transformer blocks
    for block in blocks:
        # (n_seq, n_embd) -> (n_seq, n_embd)
        x = transformer_block(x, **block, n_head=n_head)

    # projection to vocab
    x = layer_norm(x, **ln_f)  # (n_seq, n_embd) -> (n_seq, n_embd)
    logits = x @ wte.T  # (n_seq, n_embd) -> (n_seq, n_vocab)

    # reshape to (batch_size=1, sequence_length, vocab_size)
    return logits
