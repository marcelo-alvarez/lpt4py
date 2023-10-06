# Compute real to complex FFTs on GPUs with Jax on multiple GPUs.
# Adapted from version developed by Lukas Winkler, see:
#   https://gist.github.com/Findus23/eb5ecb9f65ccf13152cda7c7e521cbdd

from typing import Callable

import jax
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import PartitionSpec as P, NamedSharding


def fft_partitioner(fft_func: Callable[[jax.Array], jax.Array], partition_spec: P):
    @custom_partitioning
    def func(x):
        return fft_func(x)

    def supported_sharding(sharding, shape):
        return NamedSharding(sharding.mesh, partition_spec)

    def partition(arg_shapes, arg_shardings, result_shape, result_sharding):
        return fft_func, supported_sharding(arg_shardings[0], arg_shapes[0]), (
            supported_sharding(arg_shardings[0], arg_shapes[0]),)

    def infer_sharding_from_operands(arg_shapes, arg_shardings, shape):
        return supported_sharding(arg_shardings[0], arg_shapes[0])

    func.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition
    )
    return func


def _fft_XY(x):
    return jax.numpy.fft.fftn(x, axes=[0, 1])


def _fft_Z(x):
    return jax.numpy.fft.rfft(x, axis=2)


def _ifft_XY(x):
    return jax.numpy.fft.ifftn(x, axes=[0, 1])


def _ifft_Z(x):
    return jax.numpy.fft.irfft(x, axis=2)


fft_XY = fft_partitioner(_fft_XY, P(None, None, "gpus"))
fft_Z = fft_partitioner(_fft_Z, P(None, "gpus"))
ifft_XY = fft_partitioner(_ifft_XY, P(None, None, "gpus"))
ifft_Z = fft_partitioner(_ifft_Z, P(None, "gpus"))


def rfftn(x):
    x = fft_Z(x)
    x = fft_XY(x)
    return x


def irfftn(x):
    x = ifft_XY(x)
    x = ifft_Z(x)
    return x



