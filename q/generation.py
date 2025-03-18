from typing import Generator

import mlx.core as mx

from .gpt2 import GPT2Model


class TokenGenerator:
    model: GPT2Model

    def __init__(
        self,
        model: GPT2Model,
    ):
        self.model = model

    def __call__(
        self,
        /,
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

        Yields:
            生成されたトークン
        """
        for _ in range(max_new_tokens):  # auto-regressive decode loop
            logits = self.model(inputs).logits  # model forward pass
            next_id = mx.argmax(logits[-1])  # greedy sampling
            next_token = int(next_id)
            inputs.append(next_token)  # append prediction to input

            yield next_token
