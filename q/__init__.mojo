import math
from tensor import Tensor
from algorithm.functional import vectorize

# TODO: Where is PI defined?
alias PI = 3.14159265358979323846


fn tanh[dtype: DType](x: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Computes the hyperbolic tangent of the input tensor element-wise.
    """
    var result_tensor: Tensor[dtype] = Tensor[dtype](x.shape())
    alias simd_width = simdwidthof[dtype]()

    @parameter
    fn closure[simdwidth: Int](i: Int):
        var simd_data = x.load[width=simd_width](i)

        result_tensor.store[width=simd_width](
            i, math.tanh[dtype, simd_width](simd_data)
        )

    vectorize[closure, simd_width](x.num_elements())

    return result_tensor


fn gelu(x: Tensor[DType.float32]) raises -> Tensor[DType.float32]:
    # return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    var a = math.sqrt(2.0 / PI) * (x + 0.044715 * (x**3))
    var b = 1.0 + tanh(a)

    return 0.5 * x * b
