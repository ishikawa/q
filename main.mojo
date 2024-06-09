from tensor import Tensor
from utils.index import Index
from q.utils import max


def main():
    var t = Tensor[DType.float32](3, 3)

    t[Index(0, 0)] = 1.0
    t[Index(0, 1)] = 2.0
    t[Index(0, 2)] = 3.0

    t[Index(1, 0)] = 4.0
    t[Index(1, 1)] = 5.0
    t[Index(1, 2)] = 6.0

    t[Index(2, 0)] = 7.0
    t[Index(2, 1)] = 8.0
    t[Index(2, 2)] = 9.0

    print(t)
    # TODO: Segmentation fault
    var y = max(t)
    print(y)
