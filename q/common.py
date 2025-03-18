from typing import Any, Literal, TypeAlias, TypedDict

MODEL_SIZE = ["124M", "355M", "774M", "1558M"]
ModelSize: TypeAlias = Literal["124M", "355M", "774M", "1558M"]

# Supported backend
MODEL_BACKEND = Literal["mlx", "numpy"]


class GPT2HyperParams(TypedDict):
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int


class GPT2LayerNormParams[A](TypedDict):
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
    b: A
    # gamma
    g: A


class GPT2Params[A](TypedDict):
    wte: A
    wpe: A
    blocks: list[dict[str, Any]]  # nested dict
    ln_f: GPT2LayerNormParams[A]
