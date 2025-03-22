# The script is designed to be run from the command line. It takes a model size to download.
import argparse
import json
import os
import re
from typing import Literal

from q.common import GPT2Params
from q.params import save_params_to_safetensors


def download_gpt2_files(
    *,
    model_size: Literal["124M", "355M", "774M", "1558M"],
    model_dir: str,
    overwrite: bool = False,
):
    import requests
    from tqdm import tqdm

    assert model_size in ["124M", "355M", "774M", "1558M"]

    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath) and not overwrite:
            continue

        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(filepath, "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams) -> GPT2Params:
    import numpy as np
    import tensorflow as tf  # type: ignore

    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            if not m:
                raise ValueError(f"Unexpected name: {name}")
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params  # type: ignore


def download_encoder_hparams_and_params(
    *,
    model_size: Literal["124M", "355M", "774M", "1558M"],
    models_dir: str,
    overwrite: bool = False,
):
    import tensorflow as tf  # type: ignore

    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)
    params_safetensors_path = os.path.join(model_dir, "params.safetensors")

    # download files
    os.makedirs(model_dir, exist_ok=True)
    download_gpt2_files(model_size=model_size, model_dir=model_dir, overwrite=overwrite)

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    assert tf_ckpt_path

    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    # Serialize params to local disk to avoid loading the tf checkpoint files again
    save_params_to_safetensors(params, params_safetensors_path, overwrite=overwrite)

    # Remove the tf checkpoint files to save disk space
    for filename in os.listdir(model_dir):
        if filename.startswith(os.path.basename(tf_ckpt_path)):
            os.remove(os.path.join(model_dir, filename))


def main():
    parser = argparse.ArgumentParser(description="Main script for text generation.")
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["124M", "355M", "774M", "1558M"],
        default="124M",
        help="Size of the model to use",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory where models are stored",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing model files",
    )

    args = parser.parse_args()

    download_encoder_hparams_and_params(
        model_size=args.model_size, models_dir=args.models_dir, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
