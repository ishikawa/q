from q import gelu
import numpy as np


def test():
    gelu(np.random.randn(1000)).block_until_ready()


if __name__ == "__main__":
    import timeit

    n = 10000
    t = timeit.timeit("test()", setup="from __main__ import test", number=n)
    print(f"{t / n:.6f} s per loop")
