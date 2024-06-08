from testing import assert_almost_equal
from tensor import Tensor
from utils.index import Index
from q import gelu


def test_gleu():
    var t = Tensor[DType.float32](2, 2)

    t[Index(0, 0)] = 1.0
    t[Index(0, 1)] = 2.0

    t[Index(1, 0)] = -2.0
    t[Index(1, 1)] = 0.5

    var r = gelu(t)

    assert_almost_equal(r[Index(0, 0)], 0.841192)
    assert_almost_equal(r[Index(0, 1)], 1.954597)
    assert_almost_equal(r[Index(1, 0)], -0.045402)
    assert_almost_equal(r[Index(1, 1)], 0.345714)
