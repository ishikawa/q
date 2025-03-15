from typing import Any, Dict

import mlx.core as mx

from q.params import build_params_from_safetensors


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
