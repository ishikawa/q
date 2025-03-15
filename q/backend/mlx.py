from pprint import pprint  # noqa: F401
from typing import Any, Callable, Generator, Optional, Union

import mlx.core as mx

from q.common import GPT2Params


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


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
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
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(
    inputs: list[int],
    *,
    params: GPT2Params,
    n_head: int,
    n_tokens_to_generate: int,
    update_progress: Optional[Callable[[int, list[int]], Optional[bool]]] = None,
) -> list[int]:
    """
    トークンを生成する関数
    
    Args:
        inputs: 入力トークンのリスト
        params: モデルパラメータ
        n_head: ヘッド数
        n_tokens_to_generate: 生成するトークン数
        update_progress: 進捗更新用コールバック関数。
                          引数は (count, token_id) で、token_id は生成されたトークンID、
                          count は生成されたトークン数。
        
    Returns:
        生成されたトークンのリスト
    """
    wte: mx.array = ndarray_to_mlx_deeply(params["wte"])
    wpe: mx.array = ndarray_to_mlx_deeply(params["wpe"])
    blocks: list[dict[str, Any]] = ndarray_to_mlx_deeply(params["blocks"])
    ln_f: dict[str, Any] = ndarray_to_mlx_deeply(params["ln_f"])
    
    generated_tokens = []

    for _ in range(n_tokens_to_generate):  # auto-regressive decode loop
        logits = gpt2(inputs, wte, wpe, blocks, ln_f, n_head=n_head)
        next_id = mx.argmax(logits[-1])
        next_token = int(next_id.item())  # type: ignore
        inputs.append(next_token)  # append prediction to input
        generated_tokens.append(next_token)

        if update_progress:
            # コールバックにトークンを渡す
            update_progress(1, [next_token])
    
    return generated_tokens


def ndarray_to_mlx_deeply(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: ndarray_to_mlx_deeply(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [ndarray_to_mlx_deeply(v) for v in d]
    else:
        return mx.array(d)
