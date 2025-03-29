from typing import Generator, Optional

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
        # token ids of the input string
        inputs: list[int],
        *,
        # The maximum length the generated tokens can have. Corresponds to the
        # length of the input prompt + `max_new_tokens`. Its effect is overridden
        # by `max_new_tokens`, if also set.
        max_length: int = 40,
        # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        max_new_tokens: Optional[int] = None,
    ) -> Generator[int, None, None]:
        """
        generate method for text generation
        """
        # calculate `max_length` and `max_new_tokens`
        if max_new_tokens is None:
            if len(inputs) > max_length:
                raise ValueError(
                    f"Input length {len(inputs)} exceeds max_length {max_length}."
                )

            max_new_tokens = max_length - len(inputs)
        else:
            max_length = len(inputs) + max_new_tokens

        assert max_length is not None
        assert max_new_tokens is not None

        if max_length > self.model.hparams["n_ctx"]:
            raise ValueError(
                f"max_length {max_length} exceeds model context length {self.model.hparams['n_ctx']}."
            )

        # clone inputs to avoid modifying the original list
        inputs_array = mx.array([inputs])

        for _ in range(max_new_tokens):  # auto-regressive decode loop
            # model forward pass (shape: [1, n_seq, n_vocab])
            logits = self.model(inputs_array).logits

            # greedy sampling
            last_token_logits = logits[0, -1]  # shape: [n_vocab]
            next_id = mx.argmax(last_token_logits)

            # append prediction to input
            next_token = int(next_id)
            inputs_array = mx.concat([inputs_array, mx.array([[next_token]])], axis=1)

            yield next_token
