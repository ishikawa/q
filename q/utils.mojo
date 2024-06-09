import math
from tensor import Tensor
from algorithm.functional import vectorize

# TODO: Where is PI defined?
alias PI = 3.14159265358979323846


fn tanh[dtype: DType](x: Tensor[dtype]) -> Tensor[dtype]:
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


fn max[
    dtype: DType
](x: Tensor[dtype], *, axis: Int = -1) raises -> Tensor[dtype]:
    """
    Computes the maximum value of the input tensor.
    """
    var i = x.argmax(axis=axis)
    #    var result_tensor: Tensor[dtype] = Tensor[dtype](i.shape())

    return i
