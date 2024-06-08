from tensor import Tensor
from utils.index import Index
from q import gelu


def main():
    var t = Tensor[DType.float32](2, 2)

    t[Index(0, 0)] = 1.0
    t[Index(0, 1)] = 2.0

    t[Index(1, 0)] = -2.0
    t[Index(1, 1)] = 0.5

    print(t)
    var y = gelu(t)
    print(y)
