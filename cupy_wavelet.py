import math

import numpy as np


_WAVELET_FILTERS = {
    "db1": (
        [
            0.7071067811865476,
            0.7071067811865476,
        ],
        [
            -0.7071067811865476,
            0.7071067811865476,
        ],
    ),
    "db6": (
        [
            -0.0010773010853084796,
            0.004777257510945511,
            0.0005538422009938016,
            -0.03158203931748603,
            0.027522865530305728,
            0.09750160558732304,
            -0.12976686756726194,
            -0.22626469396543983,
            0.3152503517091982,
            0.7511339080210954,
            0.49462389039845306,
            0.11154074335008017,
        ],
        [
            -0.11154074335008017,
            0.49462389039845306,
            -0.7511339080210954,
            0.3152503517091982,
            0.22626469396543983,
            -0.12976686756726194,
            -0.09750160558732304,
            0.027522865530305728,
            0.03158203931748603,
            0.0005538422009938016,
            -0.004777257510945511,
            -0.0010773010853084796,
        ],
    ),
}


def soft_threshold(input, lam, xp):
    abs_input = xp.abs(input)
    scale = xp.maximum(1 - lam / xp.maximum(abs_input, 1e-12), 0)
    return input * scale


class CupyWavelet:
    """Decimated separable wavelet operator that keeps array math on the GPU."""

    def __init__(self, ishape, wave_name="db6", axes=None, level=None, xp=None):
        if wave_name not in _WAVELET_FILTERS:
            raise ValueError("Unsupported wavelet: {}".format(wave_name))

        self.ishape = tuple(ishape)
        self.wave_name = wave_name
        self.axes = tuple(range(len(self.ishape))) if axes is None else tuple(axes)
        self.xp = np if xp is None else xp

        filter_len = len(_WAVELET_FILTERS[wave_name][0])
        if level is None:
            level = min(_max_dwt_level(self.ishape[axis], filter_len) for axis in self.axes)
        self.level = max(0, int(level))

        self.oshape = list(self.ishape)
        multiple = 2**self.level
        for axis in self.axes:
            if multiple > 1:
                self.oshape[axis] = int(math.ceil(self.oshape[axis] / multiple) * multiple)
            elif self.oshape[axis] % 2:
                self.oshape[axis] += 1
        self.oshape = tuple(self.oshape)

    def __mul__(self, input):
        return self(input)

    def __call__(self, input):
        xp = self.xp
        output = xp.zeros(self.oshape, dtype=input.dtype)
        output[_origin_slices(self.ishape)] = input

        current_shape = list(self.oshape)
        for _ in range(self.level):
            block_slice = _block_slices(output.ndim, self.axes, current_shape)
            block = output[block_slice]
            for axis in self.axes:
                block = _dwt_axis(block, axis, self.wave_name, xp)
            output[block_slice] = block
            for axis in self.axes:
                current_shape[axis] //= 2

        return output

    @property
    def H(self):
        return _CupyWaveletAdjoint(self)


class _CupyWaveletAdjoint:
    def __init__(self, wavelet):
        self.wavelet = wavelet

    def __call__(self, input):
        xp = self.wavelet.xp
        output = input.copy()

        for level in range(self.wavelet.level, 0, -1):
            current_shape = list(self.wavelet.oshape)
            divisor = 2 ** (level - 1)
            for axis in self.wavelet.axes:
                current_shape[axis] //= divisor

            block_slice = _block_slices(output.ndim, self.wavelet.axes, current_shape)
            block = output[block_slice]
            for axis in reversed(self.wavelet.axes):
                block = _idwt_axis(block, axis, self.wavelet.wave_name, xp)
            output[block_slice] = block

        return output[_origin_slices(self.wavelet.ishape)]


def _max_dwt_level(data_len, filter_len):
    if data_len < filter_len - 1:
        return 0
    return int(math.floor(math.log(data_len / (filter_len - 1), 2)))


def _origin_slices(shape):
    return tuple(slice(0, dim) for dim in shape)


def _block_slices(ndim, axes, shape):
    slices = [slice(None)] * ndim
    for axis in axes:
        slices[axis] = slice(0, shape[axis])
    return tuple(slices)


def _filter_arrays(wave_name, xp, dtype):
    real_dtype = np.float32 if np.dtype(dtype).itemsize <= 8 else np.float64
    lo, hi = _WAVELET_FILTERS[wave_name]
    return xp.asarray(lo, dtype=real_dtype), xp.asarray(hi, dtype=real_dtype)


def _dwt_axis(input, axis, wave_name, xp):
    lo_filter, hi_filter = _filter_arrays(wave_name, xp, input.dtype)
    n = input.shape[axis]
    half = n // 2
    even = xp.arange(0, n, 2)

    out_shape = list(input.shape)
    out_shape[axis] = half
    lo = xp.zeros(out_shape, dtype=input.dtype)
    hi = xp.zeros(out_shape, dtype=input.dtype)

    for idx in range(lo_filter.size):
        shifted = xp.roll(input, -idx, axis=axis)
        samples = xp.take(shifted, even, axis=axis)
        lo = lo + lo_filter[idx] * samples
        hi = hi + hi_filter[idx] * samples

    return xp.concatenate((lo, hi), axis=axis)


def _idwt_axis(input, axis, wave_name, xp):
    lo_filter, hi_filter = _filter_arrays(wave_name, xp, input.dtype)
    n = input.shape[axis]
    half = n // 2
    lo, hi = xp.split(input, 2, axis=axis)

    output = xp.zeros_like(input)
    even_slice = [slice(None)] * input.ndim
    even_slice[axis] = slice(0, n, 2)
    even_slice = tuple(even_slice)

    for idx in range(lo_filter.size):
        up = xp.zeros_like(input)
        up[even_slice] = lo
        output = output + lo_filter[idx] * xp.roll(up, idx, axis=axis)

        up = xp.zeros_like(input)
        up[even_slice] = hi
        output = output + hi_filter[idx] * xp.roll(up, idx, axis=axis)

    return output
