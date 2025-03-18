from typing import Generator

import mlx.core as mx

from .gpt2 import GPT2Model


def generate[
    A
](
    model: GPT2Model,
    inputs: list[int],
    *,
    # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    max_new_tokens: int,
) -> Generator[int, None, None]:
    """
    トークンを生成する関数

    Args:
        inputs: 入力トークンのリスト
        max_new_tokens: 生成するトークン数

    Returns:
        生成されたトークンのリスト
    """
    for _ in range(max_new_tokens):  # auto-regressive decode loop
        logits = model(inputs).logits  # model forward pass
        next_id = mx.argmax(logits[-1])  # greedy sampling
        next_token = int(next_id)
        inputs.append(next_token)  # append prediction to input

        yield next_token
