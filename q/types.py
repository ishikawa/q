from typing import Literal, TypedDict, TypeAlias


MODEL_SIZE = ["124M", "355M", "774M", "1558M"]
ModelSize: TypeAlias = Literal["124M", "355M", "774M", "1558M"]


class HyperParameters(TypedDict):
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
