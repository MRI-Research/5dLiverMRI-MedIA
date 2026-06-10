import math
import numpy as np


# PyWavelets / PTWT Daubechies analysis filters.
# The DWT below uses the PyWavelets zero-extension convention:
#   c[k] = sum_i h[i] * x[2*k + 1 - i]
# with x[j] = 0 outside the original array.
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
            0.0005538422011614961,
            -0.03158203931748603,
            0.027522865530305727,
            0.09750160558732304,
            -0.12976686756726194,
            -0.22626469396543983,
            0.31525035170919763,
            0.7511339080210954,
            0.49462389039845306,
            0.11154074335010947,
        ],
        [
            -0.11154074335010947,
            0.49462389039845306,
            -0.7511339080210954,
            0.31525035170919763,
            0.22626469396543983,
            -0.12976686756726194,
            -0.09750160558732304,
            0.027522865530305727,
            0.03158203931748603,
            0.0005538422011614961,
            -0.004777257510945511,
            -0.0010773010853084796,
        ],
    ),
}


def soft_threshold(input, lam, xp):
    """Complex soft-thresholding used by the old media code.

    In the PDHG dual update, p - soft_threshold(p, lambda) is equivalent to
    projection of complex coefficients onto the pointwise L_inf ball:
        p <- p / max(1, |p| / lambda).
    """
    abs_input = xp.abs(input)
    scale = xp.maximum(1 - lam / xp.maximum(abs_input, 1e-12), 0)
    return input * scale


class CupyWavelet:
    """PTWT/PyWavelets-compatible zero-boundary wavelet operator.

    This class is intentionally narrow because it is meant to match the two
    operators used in the reconstruction code:

      * W1: cones-style level-1 db1 along echo axis 0, same-shaped output.
      * W2: PTWT-style wavedec3 db6 over axes (Z, Y, X), flat packed output.

    The old implementation used xp.roll(), i.e. circular boundary conditions.
    That changes the optimization problem.  This implementation uses explicit
    zero extension and the PyWavelets/PTWT coefficient lengths.
    """

    def __init__(self, ishape, wave_name="db6", axes=None, level=None, xp=None):
        if wave_name not in _WAVELET_FILTERS:
            raise ValueError("Unsupported wavelet: {}".format(wave_name))

        self.ishape = tuple(int(v) for v in ishape)
        self.wave_name = wave_name
        self.axes = tuple(range(len(self.ishape))) if axes is None else tuple(int(a) for a in axes)
        self.axes = tuple(a if a >= 0 else len(self.ishape) + a for a in self.axes)
        self.xp = np if xp is None else xp

        if self.wave_name == "db1" and len(self.axes) == 1 and self.axes[0] == 0:
            # Match cones/_w1_db1_level1_fwd for the 1-GPU case: level-1
            # pairwise Haar with zero outside the global echo range.
            if level is not None and int(level) != 1:
                raise ValueError("cones-compatible W1 db1 only supports level=1")
            self.kind = "w1_db1_level1_zero"
            self.level = 1
            self.oshape = self.ishape
            self._flat_slices = None
            return

        if self.wave_name == "db6" and len(self.axes) == 3:
            # Match ptwt.wavedec3(..., wavelet='db6', mode='zero', level=None,
            # axes=(-3,-2,-1)) applied independently to all leading dimensions.
            self.kind = "wavedec3_db6_zero_flat"
            filt_len = len(_WAVELET_FILTERS[self.wave_name][0])
            if level is None:
                level = min(_max_dwt_level(self.ishape[axis], filt_len) for axis in self.axes)
            self.level = max(0, int(level))
            self._build_wavedec3_metadata()
            self.oshape = (self._ncoeff_total,)
            return

        raise NotImplementedError(
            "This replacement intentionally implements only cones W1 "
            "(db1, axes=(0,)) and PTWT-style W2 (db6, 3 spatial axes). "
            "Got wave_name={}, axes={}.".format(self.wave_name, self.axes)
        )

    def __mul__(self, input):
        return self(input)

    def __call__(self, input):
        if self.kind == "w1_db1_level1_zero":
            return _w1_db1_level1_fwd(input, self.xp)
        if self.kind == "wavedec3_db6_zero_flat":
            return self._wavedec3_flat(input)
        raise RuntimeError("Unknown wavelet kind: {}".format(self.kind))

    @property
    def H(self):
        return _CupyWaveletAdjoint(self)

    def _build_wavedec3_metadata(self):
        shape = list(self.ishape)
        self._level_input_shapes = []   # fine -> coarse; shape before each level
        self._level_coeff_shapes = []   # fine -> coarse; shape of each band at that level
        filt_len = len(_WAVELET_FILTERS[self.wave_name][0])

        for _ in range(self.level):
            self._level_input_shapes.append(tuple(shape))
            for axis in self.axes:
                shape[axis] = _dwt_coeff_len(shape[axis], filt_len)
            self._level_coeff_shapes.append(tuple(shape))

        self._approx_shape = tuple(shape)

        # Pack order: final approximation, then details from coarse to fine.
        # Detail band order within a level is:
        #   LLH, LHL, LHH, HLL, HLH, HHL, HHH
        pack_shapes = [self._approx_shape]
        for coeff_shape in reversed(self._level_coeff_shapes):
            pack_shapes.extend([coeff_shape] * 7)

        self._pack_shapes = pack_shapes
        self._flat_slices = []
        offset = 0
        for shp in pack_shapes:
            size = int(np.prod(shp, dtype=np.int64))
            self._flat_slices.append((offset, offset + size, shp))
            offset += size
        self._ncoeff_total = int(offset)

    def _wavedec3_flat(self, input):
        xp = self.xp
        if self.level == 0:
            return input.reshape(-1).copy()

        approx = input
        details_fine_to_coarse = []
        for _ in range(self.level):
            approx, details = _dwt3_level_zero(approx, self.axes, self.wave_name, xp)
            details_fine_to_coarse.append(details)

        pieces = [approx.reshape(-1)]
        for details in reversed(details_fine_to_coarse):
            pieces.extend([band.reshape(-1) for band in details])
        return xp.concatenate(pieces, axis=0)


class _CupyWaveletAdjoint:
    def __init__(self, wavelet):
        self.wavelet = wavelet

    def __call__(self, input):
        if self.wavelet.kind == "w1_db1_level1_zero":
            return _w1_db1_level1_adj(input, self.wavelet.xp)
        if self.wavelet.kind == "wavedec3_db6_zero_flat":
            return self._waverec3_from_flat(input)
        raise RuntimeError("Unknown wavelet kind: {}".format(self.wavelet.kind))

    def _waverec3_from_flat(self, input):
        xp = self.wavelet.xp
        if self.wavelet.level == 0:
            return input.reshape(self.wavelet.ishape).copy()

        # Unpack final approximation and all detail bands.
        offset0, offset1, shp = self.wavelet._flat_slices[0]
        approx = input[offset0:offset1].reshape(shp)

        flat_idx = 1
        # Reconstruct from coarse level to fine level.
        for level_idx in range(self.wavelet.level - 1, -1, -1):
            details = []
            for _ in range(7):
                offset0, offset1, shp = self.wavelet._flat_slices[flat_idx]
                details.append(input[offset0:offset1].reshape(shp))
                flat_idx += 1

            approx = _idwt3_level_zero(
                approx,
                details,
                self.wavelet.axes,
                self.wavelet._level_input_shapes[level_idx],
                self.wavelet.wave_name,
                xp,
            )

        return approx.reshape(self.wavelet.ishape)


def _max_dwt_level(data_len, filter_len):
    # Same formula used by PyWavelets dwt_max_level.
    if data_len < filter_len - 1:
        return 0
    return int(math.floor(math.log(float(data_len) / float(filter_len - 1), 2)))


def _dwt_coeff_len(data_len, filter_len):
    # PyWavelets coefficient length for all modes except periodization,
    # including mode='zero': floor((N + L - 1) / 2).
    return (int(data_len) + int(filter_len) - 1) // 2


def _filter_arrays(wave_name, xp, dtype):
    dtype_np = np.dtype(dtype)
    if np.issubdtype(dtype_np, np.complexfloating):
        real_dtype = np.float32 if dtype_np == np.dtype(np.complex64) else np.float64
    else:
        real_dtype = np.float32 if dtype_np == np.dtype(np.float32) else np.float64
    lo, hi = _WAVELET_FILTERS[wave_name]
    return xp.asarray(lo, dtype=real_dtype), xp.asarray(hi, dtype=real_dtype)


def _mask_for_axis(valid, ndim, axis):
    shape = [1] * ndim
    shape[axis] = valid.shape[0]
    return valid.reshape(shape)


def _dwt_axis_zero(input, axis, wave_name, xp):
    """One-dimensional analysis DWT along one axis, mode='zero'.

    Matches pywt.dwt(input, wavelet, mode='zero') along the selected axis:
        c[k] = sum_i h[i] * x[2*k + 1 - i], outside samples are zero.
    """
    lo_filter, hi_filter = _filter_arrays(wave_name, xp, input.dtype)
    n = int(input.shape[axis])
    filt_len = int(lo_filter.size)
    m = _dwt_coeff_len(n, filt_len)

    out_shape = list(input.shape)
    out_shape[axis] = m
    lo = xp.zeros(out_shape, dtype=input.dtype)
    hi = xp.zeros(out_shape, dtype=input.dtype)

    k = xp.arange(m, dtype=xp.int64)
    for i in range(filt_len):
        src = 2 * k + 1 - i
        valid = (src >= 0) & (src < n)
        src_safe = xp.where(valid, src, 0)
        samples = xp.take(input, src_safe, axis=axis)
        mask = _mask_for_axis(valid.astype(samples.real.dtype), input.ndim, axis)
        lo = lo + lo_filter[i] * samples * mask
        hi = hi + hi_filter[i] * samples * mask

    return lo, hi


def _idwt_axis_zero(lo, hi, axis, out_len, wave_name, xp):
    """Adjoint/inverse of _dwt_axis_zero along one axis.

    For orthogonal db wavelets with zero extension this is the same operation
    as pywt.idwt(..., mode='zero') cropped to the requested out_len.
    """
    lo_filter, hi_filter = _filter_arrays(wave_name, xp, lo.dtype)
    m = int(lo.shape[axis])
    filt_len = int(lo_filter.size)
    out_shape = list(lo.shape)
    out_shape[axis] = int(out_len)
    output = xp.zeros(out_shape, dtype=lo.dtype)

    k_all = xp.arange(m, dtype=xp.int64)
    for i in range(filt_len):
        dst = 2 * k_all + 1 - i
        valid = (dst >= 0) & (dst < out_len)

        # Avoid zero-length advanced-indexing corner cases on CuPy.
        if int(valid.sum()) == 0:
            continue

        k_valid = k_all[valid]
        dst_valid = dst[valid]
        values = lo_filter[i] * xp.take(lo, k_valid, axis=axis)
        values = values + hi_filter[i] * xp.take(hi, k_valid, axis=axis)

        sl = [slice(None)] * output.ndim
        sl[axis] = dst_valid
        output[tuple(sl)] += values

    return output


def _dwt3_level_zero(input, axes, wave_name, xp):
    """One separable 3D DWT level over axes=(Z,Y,X).

    Returns approx and seven detail bands in order:
        LLH, LHL, LHH, HLL, HLH, HHL, HHH
    where the letters correspond to axes[0], axes[1], axes[2].
    """
    if len(axes) != 3:
        raise ValueError("_dwt3_level_zero expects exactly three axes")
    az, ay, ax = axes

    l_z, h_z = _dwt_axis_zero(input, az, wave_name, xp)

    ll_zy, lh_zy = _dwt_axis_zero(l_z, ay, wave_name, xp)
    hl_zy, hh_zy = _dwt_axis_zero(h_z, ay, wave_name, xp)

    lll, llh = _dwt_axis_zero(ll_zy, ax, wave_name, xp)
    lhl, lhh = _dwt_axis_zero(lh_zy, ax, wave_name, xp)
    hll, hlh = _dwt_axis_zero(hl_zy, ax, wave_name, xp)
    hhl, hhh = _dwt_axis_zero(hh_zy, ax, wave_name, xp)

    details = [llh, lhl, lhh, hll, hlh, hhl, hhh]
    return lll, details


def _idwt3_level_zero(approx, details, axes, original_shape, wave_name, xp):
    """Adjoint/inverse of _dwt3_level_zero for one level."""
    if len(axes) != 3:
        raise ValueError("_idwt3_level_zero expects exactly three axes")
    if len(details) != 7:
        raise ValueError("3D wavelet reconstruction expects seven detail bands")
    az, ay, ax = axes
    llh, lhl, lhh, hll, hlh, hhl, hhh = details

    x_len = int(original_shape[ax])
    y_len = int(original_shape[ay])
    z_len = int(original_shape[az])

    ll_zy = _idwt_axis_zero(approx, llh, ax, x_len, wave_name, xp)
    lh_zy = _idwt_axis_zero(lhl, lhh, ax, x_len, wave_name, xp)
    hl_zy = _idwt_axis_zero(hll, hlh, ax, x_len, wave_name, xp)
    hh_zy = _idwt_axis_zero(hhl, hhh, ax, x_len, wave_name, xp)

    l_z = _idwt_axis_zero(ll_zy, lh_zy, ay, y_len, wave_name, xp)
    h_z = _idwt_axis_zero(hl_zy, hh_zy, ay, y_len, wave_name, xp)

    output = _idwt_axis_zero(l_z, h_z, az, z_len, wave_name, xp)
    return output


def _w1_db1_level1_fwd(input, xp):
    """cones/_w1_db1_level1_fwd for the 1-GPU/full-echo case."""
    coeff = xp.zeros_like(input)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    e_len = int(input.shape[0])

    for j in range(e_len):
        if (j % 2) == 0:
            x0 = input[j]
            x1 = input[j + 1] if (j + 1 < e_len) else 0.0
            coeff[j] = (x0 + x1) * inv_sqrt2
        else:
            x0 = input[j - 1]
            x1 = input[j]
            coeff[j] = (-x0 + x1) * inv_sqrt2
    return coeff


def _w1_db1_level1_adj(input, xp):
    """cones/_w1_db1_level1_adj for the 1-GPU/full-echo case."""
    output = xp.zeros_like(input)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    e_len = int(input.shape[0])

    for j in range(e_len):
        if (j % 2) == 0:
            a = input[j]
            d_next = input[j + 1] if (j + 1 < e_len) else 0.0
            output[j] = (a - d_next) * inv_sqrt2
        else:
            a_prev = input[j - 1]
            d = input[j]
            output[j] = (a_prev + d) * inv_sqrt2
    return output
