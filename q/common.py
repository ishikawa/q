from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np

MODEL_SIZE = ["124M", "355M", "774M", "1558M"]
ModelSize: TypeAlias = Literal["124M", "355M", "774M", "1558M"]


class GPT2LayerNormParams(TypedDict):
    """
    GPT-2におけるln_fは、モデルの最終出力に適用されるLayer Normalization（層正規化）層を指します。

    この層は、モデル全体の安定性と性能を向上させるために、出力の正規化を行います。具体的には、GPT-2
    のアーキテクチャでは、各トランスフォーマーブロック内にln_1とln_2という2つのLayer Normalization層
    が存在し、さらにモデル全体の出力に対してln_fが適用されます。これにより、各層の出力が適切に正規化
    され、学習の安定性とモデルの性能が向上します。​
    """

    # beta
    b: np.ndarray
    # gamma
    g: np.ndarray


class GPT2Params(TypedDict):
    wte: np.ndarray
    wpe: np.ndarray
    blocks: list[dict[str, Any]]  # nested dict
    ln_f: GPT2LayerNormParams
