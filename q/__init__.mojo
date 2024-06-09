import math
from tensor import Tensor
from .utils import tanh, PI


fn gelu[dtype: DType](x: Tensor[dtype]) raises -> Tensor[dtype]:
    # return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    var a = math.sqrt(2.0 / PI) * (x + 0.044715 * x**3)
    var b = 1.0 + tanh(a)

    return 0.5 * x * b
