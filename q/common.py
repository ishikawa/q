from typing import Any, Literal, TypeAlias, TypedDict

import mlx.core as mx

MODEL_SIZE = ["124M", "355M", "774M", "1558M"]
ModelSize: TypeAlias = Literal["124M", "355M", "774M", "1558M"]


class GPT2HyperParams(TypedDict):
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int


class GPT2LayerNormParams(TypedDict):
    """
    In GPT-2, **ln_f** refers to the Layer Normalization (LN) layer applied to the model's final
    output.

    This layer normalizes the output to enhance the overall stability and performance of the model.
    Specifically, in the GPT-2 architecture, each transformer block contains two Layer Normalization
    layers, **ln_1** and **ln_2**. Additionally, **ln_f** is applied to the model's overall output.

    This ensures that the outputs of each layer are properly normalized, contributing to improved
    training stability and model performance.
    """

    # beta
    b: mx.array
    # gamma
    g: mx.array


class GPT2Params(TypedDict):
    wte: mx.array
    wpe: mx.array
    blocks: list[dict[str, Any]]  # nested dict
    ln_f: GPT2LayerNormParams
