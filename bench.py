from q import gelu


import jax
from jax import random

jax.config.update("jax_platform_name", "cpu")

key = random.key(1701)
x = random.normal(key, (1_000,))
gelu(x).block_until_ready()

# -- NumPy
# import numpy as np
# x = np.random.randn(1_000)


def test():
    gelu(x).block_until_ready()


if __name__ == "__main__":
    import timeit

    n = 10000
    t = timeit.timeit("test()", setup="from __main__ import test", number=n)
    print(f"{t / n:.8f} s per loop")
