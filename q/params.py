import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from mlx.core import load as load_safetensors_mlx
from safetensors.numpy import save_file

from .common import MODEL_SIZE, GPT2HyperParams, GPT2Params, ModelSize


def load_hparams_and_params(
    *,
    model_size: ModelSize,
    models_dir: str,
) -> tuple[GPT2HyperParams, GPT2Params]:
    assert model_size in MODEL_SIZE

    target_dir = os.path.join(models_dir, model_size)

    # Error when no model exists
    if not os.path.exists(target_dir):
        raise FileNotFoundError(
            f"Model {model_size} not found in {models_dir}. You need to download it first."
        )

    hparams: GPT2HyperParams = json.load(open(os.path.join(target_dir, "hparams.json")))

    # Load params.pkl or combine separate files
    params_safetensors_path = os.path.join(target_dir, "params.safetensors")
    params_safetensors_pattern = os.path.join(
        target_dir, "params_safetensors_{part_number:03d}"
    )

    # Load from single file
    if os.path.exists(params_safetensors_path):
        params = load_safetensors_mlx(params_safetensors_path, format="safetensors")
        return hparams, build_params_from_safetensors(params)

    # Load from multiple files
    if os.path.exists(params_safetensors_pattern.format(part_number=0)):
        raw_data = b""
        for part_number in range(1000):
            part_path = params_safetensors_pattern.format(part_number=part_number)
            if not os.path.exists(part_path):
                break

            with open(part_path, "rb") as f:
                raw_data += f.read()
    else:
        raise FileNotFoundError(
            f"params.safetensors or params_safetensors_nnn not found in {target_dir}. You need to download it first."
        )

    # mlx.core.load takes file path, not raw data
    # so we need to write the raw data to a temporary file first
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(raw_data)
        params = load_safetensors_mlx(f.name, format="safetensors")

    return hparams, build_params_from_safetensors(params)


def save_params_to_safetensors(
    params: GPT2Params, output_path: Union[str, Path], overwrite: bool = False
) -> None:
    """
    GPT-2のパラメータをsafetensors形式で保存する関数

    Args:
        params: GPT-2のパラメータ
        output_path: 出力先のパス
        overwrite: 既存のファイルを上書きするかどうか

    Raises:
        FileExistsError: 出力先のファイルが既に存在し、overwriteがFalseの場合
    """
    output_path = Path(output_path)

    # 出力先のディレクトリが存在しなければ作成
    os.makedirs(output_path.parent, exist_ok=True)

    # ファイルが既に存在し、上書きしない場合はエラー
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File {output_path} already exists and overwrite=False")

    # パラメータを平坦化してnumpy配列に変換
    flat_params: Dict[str, np.ndarray] = {}

    # 埋め込み層とポジショナル埋め込み
    flat_params["wte"] = np.array(params["wte"])
    flat_params["wpe"] = np.array(params["wpe"])

    # トランスフォーマーブロック
    for i, block in enumerate(params["blocks"]):
        # 自己注意層
        flat_params[f"blocks.{i}.ln_1.g"] = np.array(block["ln_1"]["g"])
        flat_params[f"blocks.{i}.ln_1.b"] = np.array(block["ln_1"]["b"])
        flat_params[f"blocks.{i}.attn.c_attn.w"] = np.array(
            block["attn"]["c_attn"]["w"]
        )
        flat_params[f"blocks.{i}.attn.c_attn.b"] = np.array(
            block["attn"]["c_attn"]["b"]
        )
        flat_params[f"blocks.{i}.attn.c_proj.w"] = np.array(
            block["attn"]["c_proj"]["w"]
        )
        flat_params[f"blocks.{i}.attn.c_proj.b"] = np.array(
            block["attn"]["c_proj"]["b"]
        )

        # フィードフォワード層
        flat_params[f"blocks.{i}.ln_2.g"] = np.array(block["ln_2"]["g"])
        flat_params[f"blocks.{i}.ln_2.b"] = np.array(block["ln_2"]["b"])
        flat_params[f"blocks.{i}.mlp.c_fc.w"] = np.array(block["mlp"]["c_fc"]["w"])
        flat_params[f"blocks.{i}.mlp.c_fc.b"] = np.array(block["mlp"]["c_fc"]["b"])
        flat_params[f"blocks.{i}.mlp.c_proj.w"] = np.array(block["mlp"]["c_proj"]["w"])
        flat_params[f"blocks.{i}.mlp.c_proj.b"] = np.array(block["mlp"]["c_proj"]["b"])

    # 最終層の正規化
    flat_params["ln_f.g"] = np.array(params["ln_f"]["g"])
    flat_params["ln_f.b"] = np.array(params["ln_f"]["b"])

    # safetensors形式で保存
    save_file(flat_params, str(output_path))


def build_params_from_safetensors(tensors: dict[str, Any]) -> GPT2Params:
    """
    safetensors形式のファイルから読み込んだ辞書からGPT-2のパラメータを構築する

    Args:
        tensors: 読み込んだsafetensors形式の辞書

    Returns:
        GPT-2のパラメータ
    """
    # GPT-2パラメータの構造に変換
    params: Dict[str, Any] = {}
    blocks = []

    # 埋め込み層
    params["wte"] = tensors["wte"]
    params["wpe"] = tensors["wpe"]

    # ブロックの数を特定
    block_indices = set()
    for key in tensors.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1:
                block_indices.add(int(parts[1]))

    # トランスフォーマーブロックを構築
    for i in sorted(block_indices):
        block = {
            "ln_1": {
                "g": tensors[f"blocks.{i}.ln_1.g"],
                "b": tensors[f"blocks.{i}.ln_1.b"],
            },
            "attn": {
                "c_attn": {
                    "w": tensors[f"blocks.{i}.attn.c_attn.w"],
                    "b": tensors[f"blocks.{i}.attn.c_attn.b"],
                },
                "c_proj": {
                    "w": tensors[f"blocks.{i}.attn.c_proj.w"],
                    "b": tensors[f"blocks.{i}.attn.c_proj.b"],
                },
            },
            "ln_2": {
                "g": tensors[f"blocks.{i}.ln_2.g"],
                "b": tensors[f"blocks.{i}.ln_2.b"],
            },
            "mlp": {
                "c_fc": {
                    "w": tensors[f"blocks.{i}.mlp.c_fc.w"],
                    "b": tensors[f"blocks.{i}.mlp.c_fc.b"],
                },
                "c_proj": {
                    "w": tensors[f"blocks.{i}.mlp.c_proj.w"],
                    "b": tensors[f"blocks.{i}.mlp.c_proj.b"],
                },
            },
        }
        blocks.append(block)

    params["blocks"] = blocks

    # 最終層の正規化
    params["ln_f"] = {
        "g": tensors["ln_f.g"],
        "b": tensors["ln_f.b"],
    }

    return params  # type: ignore
