from q.encoder import load_encoder


def test_load_encoder(model_dir):
    encoder = load_encoder("124M", model_dir)
    assert encoder is not None


def test_encode_encode(model_dir):
    encoder = load_encoder("124M", model_dir)
    text = "Hello, world!"
    tokens = encoder.encode(text)
    assert tokens is not None
    assert tokens == encoder.encode(text)
