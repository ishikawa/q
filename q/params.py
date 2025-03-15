import json
import os
import pickle
from typing import TypedDict

from .common import MODEL_SIZE, GPT2Params, ModelSize


class HyperParameters(TypedDict):
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int


def load_hparams_and_params(
    model_size: ModelSize, models_dir: str
) -> tuple[HyperParameters, GPT2Params]:
    assert model_size in MODEL_SIZE

    target_dir = os.path.join(models_dir, model_size)

    # Error when no model exists
    if not os.path.exists(target_dir):
        raise FileNotFoundError(
            f"Model {model_size} not found in {models_dir}. You need to download it first."
        )

    hparams: HyperParameters = json.load(open(os.path.join(target_dir, "hparams.json")))

    # Load params.pkl or combine separate files
    params_pkl_path = os.path.join(target_dir, "params.pkl")
    params_pkl_pattern = os.path.join(target_dir, "params_pkl_{part_number:03d}")

    if os.path.exists(params_pkl_path):
        with open(params_pkl_path, "rb") as f:
            params = pickle.load(f)
            return hparams, params
    elif os.path.exists(params_pkl_pattern.format(part_number=0)):
        data = b""
        for part_number in range(1000):
            part_path = params_pkl_pattern.format(part_number=part_number)
            if not os.path.exists(part_path):
                break

            with open(part_path, "rb") as f:
                data += f.read()

        params = pickle.loads(data)
        return hparams, params
    else:
        raise FileNotFoundError(
            f"params.pkl or params_pkl_nnn not found in {target_dir}. You need to download it first."
        )
