import mlx.core as mx
import numpy as np
import pytest

from q.eval import QLM, pad_and_concat


@pytest.fixture
def qlm():
    return QLM()


def test_eot_token_id(qlm):
    """Test that the eot_token_id property returns None as expected."""
    assert qlm.eot_token_id is None


def test_tok_encode(qlm):
    """Test that the tok_encode method correctly tokenizes a string using the model's encoder."""
    text = "Hello, world!"

    tokens = qlm.tok_encode(text)
    assert tokens is not None
    assert tokens == qlm.encoder.encode(text)


def test_pad_and_concat_right():
    """Test that pad_and_concat correctly pads and concatenates arrays with right padding."""
    # Create test arrays of different lengths
    array1 = mx.array([1, 2, 3], dtype=mx.int32)
    array2 = mx.array([4, 5], dtype=mx.int32)
    array3 = mx.array([6, 7, 8, 9], dtype=mx.int32)

    # Max length is 4 (length of array3)
    result = pad_and_concat(4, [array1, array2, array3])

    # Expected result:
    # array1 padded: [1, 2, 3, 0]
    # array2 padded: [4, 5, 0, 0]
    # array3 unchanged: [6, 7, 8, 9]
    expected = mx.array([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]], dtype=mx.int32)

    assert result.shape == (3, 4)
    assert np.array_equal(result, expected)


def test_pad_and_concat_left():
    """Test that pad_and_concat correctly pads and concatenates arrays with left padding."""
    # Create test arrays of different lengths
    array1 = mx.array([1, 2, 3], dtype=mx.int32)
    array2 = mx.array([4, 5], dtype=mx.int32)
    array3 = mx.array([6, 7, 8, 9], dtype=mx.int32)

    # Max length is 4 (length of array3)
    result = pad_and_concat(4, [array1, array2, array3], padding_side="left")

    # Expected result:
    # array1 padded: [0, 1, 2, 3]
    # array2 padded: [0, 0, 4, 5]
    # array3 unchanged: [6, 7, 8, 9]
    expected = mx.array([[0, 1, 2, 3], [0, 0, 4, 5], [6, 7, 8, 9]], dtype=mx.int32)

    assert result.shape == (3, 4)
    assert np.array_equal(result, expected)


def test_pad_and_concat_2d_input():
    """Test that pad_and_concat correctly handles 2D input arrays."""
    # Create 2D test arrays (with a batch dimension of 1)
    array1 = mx.expand_dims(mx.array([1, 2, 3], dtype=mx.int32), axis=0)
    array2 = mx.expand_dims(mx.array([4, 5], dtype=mx.int32), axis=0)

    # Max length is 3 (length of array1)
    result = pad_and_concat(3, [array1, array2])

    # Expected result:
    # array1 unchanged: [1, 2, 3]
    # array2 padded: [4, 5, 0]
    expected = mx.array([[1, 2, 3], [4, 5, 0]], dtype=mx.int32)

    assert result.shape == (2, 3)
    assert np.array_equal(result, expected)
