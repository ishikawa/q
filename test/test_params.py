from typing import Any, Dict
import os
import tempfile
import mlx.core as mx
import pytest
from safetensors.numpy import load_file
from q.params import build_params_from_safetensors, save_params_to_safetensors
from q.common import GPT2Params


def test_build_params_from_safetensors() -> None:
    """safetensors形式のデータからGPT-2パラメータを正しく構築できることを確認するテスト"""
    # テスト用のモックデータ
    mock_tensors: Dict[str, Any] = {
        # 埋め込み層
        "wte": mx.zeros((10, 20)),
        "wpe": mx.zeros((30, 20)),
        # ブロック0のパラメータ
        "blocks.0.ln_1.g": mx.ones(20),
        "blocks.0.ln_1.b": mx.zeros(20),
        "blocks.0.attn.c_attn.w": mx.zeros((20, 60)),
        "blocks.0.attn.c_attn.b": mx.zeros(60),
        "blocks.0.attn.c_proj.w": mx.zeros((20, 20)),
        "blocks.0.attn.c_proj.b": mx.zeros(20),
        "blocks.0.ln_2.g": mx.ones(20),
        "blocks.0.ln_2.b": mx.zeros(20),
        "blocks.0.mlp.c_fc.w": mx.zeros((20, 80)),
        "blocks.0.mlp.c_fc.b": mx.zeros(80),
        "blocks.0.mlp.c_proj.w": mx.zeros((80, 20)),
        "blocks.0.mlp.c_proj.b": mx.zeros(20),
        # ブロック1のパラメータ
        "blocks.1.ln_1.g": mx.ones(20),
        "blocks.1.ln_1.b": mx.zeros(20),
        "blocks.1.attn.c_attn.w": mx.zeros((20, 60)),
        "blocks.1.attn.c_attn.b": mx.zeros(60),
        "blocks.1.attn.c_proj.w": mx.zeros((20, 20)),
        "blocks.1.attn.c_proj.b": mx.zeros(20),
        "blocks.1.ln_2.g": mx.ones(20),
        "blocks.1.ln_2.b": mx.zeros(20),
        "blocks.1.mlp.c_fc.w": mx.zeros((20, 80)),
        "blocks.1.mlp.c_fc.b": mx.zeros(80),
        "blocks.1.mlp.c_proj.w": mx.zeros((80, 20)),
        "blocks.1.mlp.c_proj.b": mx.zeros(20),
        # 最終層の正規化
        "ln_f.g": mx.ones(20),
        "ln_f.b": mx.zeros(20),
    }
    # 関数を実行
    params = build_params_from_safetensors(mock_tensors)
    # 期待される結果の検証
    # 1. 返り値がGPT2Params型であることを確認
    assert isinstance(params, dict)
    # 2. 必要なキーが含まれていることを確認
    assert "wte" in params
    assert "wpe" in params
    assert "blocks" in params
    assert "ln_f" in params
    # 3. blocksの数が正しいことを確認
    assert len(params["blocks"]) == 2
    # 4. ブロック0の構造が正しいことを確認
    block0 = params["blocks"][0]
    assert "ln_1" in block0
    assert "attn" in block0
    assert "ln_2" in block0
    assert "mlp" in block0
    # 5. 最終層の正規化パラメータが正しいことを確認
    assert "g" in params["ln_f"]
    assert "b" in params["ln_f"]
    assert mx.array_equal(params["ln_f"]["g"], mock_tensors["ln_f.g"])
    assert mx.array_equal(params["ln_f"]["b"], mock_tensors["ln_f.b"])
    # 6. 特定のブロックの値が正しく設定されていることを確認
    assert mx.array_equal(
        params["blocks"][1]["ln_1"]["g"], mock_tensors["blocks.1.ln_1.g"]
    )
    assert mx.array_equal(
        params["blocks"][1]["mlp"]["c_proj"]["b"], mock_tensors["blocks.1.mlp.c_proj.b"]
    )


def test_save_params_to_safetensors() -> None:
    """GPT-2のパラメータをsafetensors形式で保存できることを確認するテスト"""
    # テスト用のGPT-2パラメータの作成
    mock_params: GPT2Params = {
        # 埋め込み層
        "wte": mx.zeros((10, 20)),
        "wpe": mx.zeros((30, 20)),
        # トランスフォーマーブロック
        "blocks": [
            {
                "ln_1": {
                    "g": mx.ones(20),
                    "b": mx.zeros(20),
                },
                "attn": {
                    "c_attn": {
                        "w": mx.zeros((20, 60)),
                        "b": mx.zeros(60),
                    },
                    "c_proj": {
                        "w": mx.zeros((20, 20)),
                        "b": mx.zeros(20),
                    },
                },
                "ln_2": {
                    "g": mx.ones(20),
                    "b": mx.zeros(20),
                },
                "mlp": {
                    "c_fc": {
                        "w": mx.zeros((20, 80)),
                        "b": mx.zeros(80),
                    },
                    "c_proj": {
                        "w": mx.zeros((80, 20)),
                        "b": mx.zeros(20),
                    },
                },
            },
            {
                "ln_1": {
                    "g": mx.ones(20),
                    "b": mx.zeros(20),
                },
                "attn": {
                    "c_attn": {
                        "w": mx.zeros((20, 60)),
                        "b": mx.zeros(60),
                    },
                    "c_proj": {
                        "w": mx.zeros((20, 20)),
                        "b": mx.zeros(20),
                    },
                },
                "ln_2": {
                    "g": mx.ones(20),
                    "b": mx.zeros(20),
                },
                "mlp": {
                    "c_fc": {
                        "w": mx.zeros((20, 80)),
                        "b": mx.zeros(80),
                    },
                    "c_proj": {
                        "w": mx.zeros((80, 20)),
                        "b": mx.zeros(20),
                    },
                },
            },
        ],
        # 最終層の正規化
        "ln_f": {
            "g": mx.ones(20),
            "b": mx.zeros(20),
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "test.safetensors")
        try:
            # パラメータの保存
            save_params_to_safetensors(mock_params, temp_path)

            # 保存したファイルが存在することを確認
            assert os.path.exists(temp_path)

            # safetensors.numpy.load_fileでファイルを読み込む
            loaded_tensors = load_file(temp_path)
            loaded_params = build_params_from_safetensors(loaded_tensors)

            # 元のパラメータと読み込んだパラメータの構造を比較
            assert set(loaded_params.keys()) == set(mock_params.keys())
            assert len(loaded_params["blocks"]) == len(mock_params["blocks"])

            # いくつかの重要なパラメータの値を比較
            assert mx.array_equal(loaded_params["wte"], mock_params["wte"])
            assert mx.array_equal(loaded_params["wpe"], mock_params["wpe"])
            assert mx.array_equal(loaded_params["ln_f"]["g"], mock_params["ln_f"]["g"])
            assert mx.array_equal(loaded_params["ln_f"]["b"], mock_params["ln_f"]["b"])

            # ブロックの特定のパラメータを比較
            assert mx.array_equal(
                loaded_params["blocks"][0]["ln_1"]["g"],
                mock_params["blocks"][0]["ln_1"]["g"],
            )
            assert mx.array_equal(
                loaded_params["blocks"][1]["mlp"]["c_proj"]["w"],
                mock_params["blocks"][1]["mlp"]["c_proj"]["w"],
            )

        finally:
            # テスト終了後、一時ファイルを削除
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def test_save_params_to_safetensors_overwrite() -> None:
    """GPT-2のパラメータをsafetensors形式で保存時のoverwriteオプションのテスト"""
    # シンプルなテスト用パラメータ
    mock_params: GPT2Params = {
        "wte": mx.zeros((2, 4)),
        "wpe": mx.zeros((2, 4)),
        "blocks": [],
        "ln_f": {
            "g": mx.ones(4),
            "b": mx.zeros(4),
        },
    }

    # 一時ファイルを使ってパラメータを保存
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "test.safetensors")

        try:
            # 最初の保存
            save_params_to_safetensors(mock_params, temp_path)

            # overwrite=Falseで保存を試みるとFileExistsErrorが発生することを確認
            with pytest.raises(FileExistsError):
                save_params_to_safetensors(mock_params, temp_path, overwrite=False)

            # overwrite=Trueで保存すると成功することを確認
            save_params_to_safetensors(mock_params, temp_path, overwrite=True)

        finally:
            # テスト終了後、一時ファイルを削除
            if os.path.exists(temp_path):
                os.unlink(temp_path)
