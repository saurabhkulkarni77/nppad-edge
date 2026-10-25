"""MultiRocket + HYDRA feature transforms in pure NumPy.

These are faithful re-implementations of the forward transforms of
aeon 1.6.0's ``MultiRocket`` and ``HydraTransformer`` (the exact
implementations used by the Homework 2 SOTA study), driven entirely by
kernel parameters extracted from the fitted aeon objects at export
time. The edge runtime therefore needs no aeon / torch / numba — only
NumPy. Equivalence with aeon is verified numerically at export time
(see train.py) and in the test-suite.

MultiRocket (aeon semantics): 84 fixed base kernels of length 9 with
weights in {-1, +2} (an all -1 kernel plus +3 at three tap positions
given by ``indices``). Each kernel is combined with a dilation and a
random subset of channels; for every (dilation, kernel) combination a
set of bias thresholds is fitted on the training data, and four pooled
features are produced per bias: PPV (proportion of positives), longest
stretch below the bias, mean of (value + bias) over positives, and mean
index of positives. Features are computed on the series and on its
first difference. Short series are zero-padded (in scaled space) to
length 12 first — the study's own fix for aeon's minimum length.

HYDRA (aeon semantics): groups of competing kernels (length 9, weights
drawn at fit time, normalised to zero mean / unit L1). Channels are
summed over a random subset per group, convolved per group, and for
each time step the winning (max) kernel accumulates the max value
("soft" count) while the losing (min) kernel accumulates one ("hard"
count). Computed on the series and its first difference.
"""
from __future__ import annotations

import numpy as np

PAD_TO = 12          # study fix: pad scaled windows to >= 12 steps
KERNEL_LENGTH = 9


def pad_windows(X, pad_to=PAD_TO):
    """Zero-pad (n, C, T) series at the end to ``pad_to`` steps."""
    T = X.shape[2]
    if T >= pad_to:
        return X
    return np.pad(X, ((0, 0), (0, 0), (0, pad_to - T)))


# ---------------------------------------------------------------------
# MultiRocket
# ---------------------------------------------------------------------

def _alpha_gamma(X, dilation, ref_T=None):
    """aeon's C_alpha / C_gamma construction for one example, replicated
    operation-for-operation (float32, same accumulation order).

    The fitted biases are quantiles of these very convolution outputs,
    so exact ties at the bias are common — and whether an output counts
    as "above bias" then depends on float32 rounding, i.e. on the
    summation order. aeon builds C_alpha by starting from A = -X and
    adding shifted copies of A in the order: base, -4d, -3d, -2d, -d,
    +d, +2d, +3d, +4d (sliced in-place adds, zero-padding semantics).
    C_gamma[g] is a zero array with one shifted copy of G = 3X assigned
    into the matching slice. This function performs the identical
    operations so the resulting C matches aeon's bit-for-bit.

    ``ref_T``: aeon's first-difference section computes its slice
    bounds from the RAW series length (n_timepoints) while its arrays
    hold the differenced series (length T-1) — an off-by-one in aeon
    that shifts the negative taps by one sample. We replicate it
    faithfully: pass ref_T = raw length for the diff half.
    """
    C, T = X.shape
    if ref_T is None:
        ref_T = T
    A = -X.astype(np.float32)                       # alpha * X, alpha = -1
    G = (np.float32(3.0) * X).astype(np.float32)    # gamma * X, gamma = 3
    padding = ((KERNEL_LENGTH - 1) * dilation) // 2

    C_alpha = A.copy()
    C_gamma = np.zeros((KERNEL_LENGTH, C, T), dtype=np.float32)
    C_gamma[KERNEL_LENGTH // 2] = G

    # gamma_index 0..3: add A shifted into the tail slices; the slice
    # bounds come from ref_T (== T for the raw half, T+1 for diff)
    end = ref_T - padding
    for gamma_index in range(KERNEL_LENGTH // 2):
        if end > 0:
            C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
            C_gamma[gamma_index, :, -end:] = G[:, :end]
        end += dilation
    # gamma_index 5..8: add A shifted by +(g-4)*d into the head slices
    start = dilation
    for gamma_index in range(KERNEL_LENGTH // 2 + 1, KERNEL_LENGTH):
        if start < T:
            C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
            C_gamma[gamma_index, :, :-start] = G[:, start:]
        start += dilation
    return C_alpha, C_gamma


def _bias_features(C, biases):
    """aeon's four pooled features for one convolution output.

    C: (T,) convolution output. biases: (B,) thresholds.
    Returns (B, 4): [ppv, max_stretch, mean_excess, mean_index].
    An empty C (heavy trim) yields [nan, 0, 0, -1] exactly as aeon's
    numba code does before its final nan_to_num.
    """
    B = len(biases)
    T = len(C)
    out = np.empty((B, 4), dtype=np.float64)
    if T == 0:
        out[:, 0] = np.nan
        out[:, 1] = 0.0
        out[:, 2] = 0.0
        out[:, 3] = -1.0
        return out
    Cb = C[None, :].astype(np.float32)   # (1, T)
    bb = biases[:, None].astype(np.float32)   # (B, 1)
    above = Cb > bb                      # strictly greater, as aeon
    below = Cb < bb                      # strictly less, as aeon
    ppv = above.sum(axis=1)
    out[:, 0] = ppv / T
    # longest "stretch": aeon tracks j - last_below_index at each
    # below-bias position (and T-1-last at the end), taking the max.
    max_stretch = np.zeros(B, dtype=np.float64)
    last_val = np.zeros(B, dtype=np.int64)
    for j in range(T):
        bj = below[:, j]
        stretch = np.where(bj, j - last_val, -np.inf)
        max_stretch = np.maximum(max_stretch, stretch)
        last_val = np.where(bj, j, last_val)
    tail = (T - 1) - last_val
    max_stretch = np.maximum(max_stretch, tail)
    out[:, 1] = max_stretch
    excess = np.where(above, Cb + bb, np.float32(0)).sum(axis=1, dtype=np.float32)
    idx_sum = np.where(above, np.arange(T)[None, :], 0).sum(axis=1)
    safe = ppv > 0
    out[:, 2] = np.where(safe, excess / np.maximum(ppv, 1), 0.0)
    out[:, 3] = np.where(safe, idx_sum / np.maximum(ppv, 1), -1.0)
    return out


class MultiRocketFeatures:
    """NumPy forward transform replicating aeon's MultiRocket."""

    def __init__(self, indices, parameter, parameter1):
        """indices: (84, 3) tap triples. parameter / parameter1: tuples
        (n_channels_per_combination, channel_indices, dilations,
        n_features_per_dilation, biases) for raw / first-difference.

        NOTE (aeon semantics, verified from source): the diff half
        reuses the RAW half's channel subsets — only its dilations,
        feature counts and biases come from parameter1."""
        self.indices = np.asarray(indices)
        self.parameter = parameter
        self.parameter1 = (parameter[0], parameter[1],
                           parameter1[2], parameter1[3], parameter1[4])
        nf = len(self.indices) * int(np.sum(parameter[3]))
        nf1 = len(self.indices) * int(np.sum(parameter1[3]))
        self.n_features_ = (nf + nf1) * 4
        self._nf = nf
        self._nf1 = nf1

    def _half_into(self, X, params, out, offset, ref_T=None):
        """Write one half (raw or first-difference) into ``out`` at
        ``offset``. The four pooled features for slot i land at
        offset + i, + i + nf, + i + 2nf, + i + 3nf where nf is the
        RAW half's feature count — exactly aeon's indexing."""
        (n_chan_per, chan_idx, dilations, n_fpd, biases) = params
        nf = self._nf
        combination_index = 0
        n_channels_start = 0
        feature_index_start = 0
        for dilation_index, dilation in enumerate(dilations):
            dilation = int(dilation)
            padding = ((KERNEL_LENGTH - 1) * dilation) // 2
            n_feat_this = int(n_fpd[dilation_index])
            A, G = _alpha_gamma(X, dilation, ref_T=ref_T)
            for kernel_index in range(len(self.indices)):
                n_ch = int(n_chan_per[combination_index])
                channels = chan_idx[n_channels_start:n_channels_start + n_ch]
                n_channels_start += n_ch
                i0, i1, i2 = (int(t) for t in self.indices[kernel_index])
                M = (A[channels] + G[i0, channels] + G[i1, channels]
                     + G[i2, channels])
                C = M[0].copy()          # sequential float32 channel sum,
                for _r in range(1, M.shape[0]):   # as in aeon's numba code
                    C += M[_r]
                _padding1 = (dilation_index % 2 + kernel_index) % 2
                if _padding1 == 1 and padding > 0:
                    C = C[padding:-padding] if padding * 2 < len(C) else C[:0]
                b = biases[feature_index_start:feature_index_start + n_feat_this]
                bf = _bias_features(C, b.astype(np.float32))
                idx = offset + np.arange(feature_index_start,
                                         feature_index_start + n_feat_this)
                out[idx] = bf[:, 0]
                out[idx + nf] = bf[:, 1]
                out[idx + 2 * nf] = bf[:, 2]
                out[idx + 3 * nf] = bf[:, 3]
                feature_index_start += n_feat_this
                combination_index += 1

    def transform_one(self, x):
        """x: (C, T) float32 single window (already padded)."""
        out = np.zeros(self.n_features_, dtype=np.float64)
        xf = x.astype(np.float32)
        self._half_into(xf, self.parameter, out, 0)
        X1 = np.diff(xf, axis=1)
        # aeon's diff section derives slice bounds from the raw length
        self._half_into(X1, self.parameter1, out, self.n_features_ // 2,
                        ref_T=xf.shape[1])
        return np.nan_to_num(out).astype(np.float32)

    def transform(self, X):
        """X: (n, C, T) -> (n, n_features_) float32."""
        return np.stack([self.transform_one(X[i]) for i in range(len(X))])


# ---------------------------------------------------------------------
# HYDRA
# ---------------------------------------------------------------------

def _conv1d_same(x, w, dilation, padding):
    """1D cross-correlation, torch F.conv1d semantics (no flip),
    zero padding, output length = len(x) + 2*padding - dilation*8."""
    xp = np.pad(x, (padding, padding)) if padding > 0 else x
    T_out = len(xp) - dilation * (len(w) - 1)
    if T_out <= 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(T_out, dtype=np.float64)
    for k in range(len(w)):
        out += w[k] * xp[k * dilation: k * dilation + T_out]
    return out


class HydraFeatures:
    """NumPy forward transform replicating aeon's HydraTransformer."""

    def __init__(self, W, idx, dilations, paddings, n_groups, n_kernels):
        """W: (n_dilations, divisor, k*h, 9) kernel weights.
        idx: (n_dilations, divisor, h, n_channels_per) channel subsets."""
        self.W = np.asarray(W, dtype=np.float64)
        self.idx = np.asarray(idx)
        self.dilations = [int(d) for d in dilations]
        self.paddings = [int(p) for p in paddings]
        self.g = int(n_groups)
        self.k = int(n_kernels)
        self.divisor = int(self.W.shape[1])
        self.h = self.g // self.divisor
        self.n_features_ = len(self.dilations) * self.divisor * 2 * self.h * self.k

    def transform_one(self, x):
        """x: (C, T) float32 -> (n_features_,) float32."""
        feats = []
        diff_x = np.diff(x, axis=1)
        for di, d in enumerate(self.dilations):
            p = self.paddings[di]
            for diff_index in range(self.divisor):
                src = x if diff_index == 0 else diff_x
                # sum the selected channels per group: (h, T)
                sel = self.idx[di, diff_index]                    # (h, ncp)
                S = src[sel].sum(axis=1)                          # (h, T)
                Wg = self.W[di, diff_index]                       # (k*h, 9)
                count_max = np.zeros((self.h, self.k), dtype=np.float64)
                count_min = np.zeros((self.h, self.k), dtype=np.float64)
                for g in range(self.h):
                    kernels = Wg[g * self.k:(g + 1) * self.k]     # (k, 9)
                    Z = np.stack([_conv1d_same(S[g], kernels[j], d, p)
                                  for j in range(self.k)])        # (k, T_out)
                    if Z.shape[1] == 0:
                        continue
                    amax = np.argmax(Z, axis=0)
                    vmax = Z[amax, np.arange(Z.shape[1])]
                    np.add.at(count_max[g], amax, vmax)
                    amin = np.argmin(Z, axis=0)
                    np.add.at(count_min[g], amin, 1.0)
                feats.append(count_max.ravel())
                feats.append(count_min.ravel())
        return np.concatenate(feats).astype(np.float32)

    def transform(self, X):
        return np.stack([self.transform_one(X[i]) for i in range(len(X))])


class SparseScaler:
    """aeon's _SparseScaler for HYDRA features, in NumPy."""

    def fit(self, X):
        Xs = np.sqrt(np.clip(X, 0, None))
        self.epsilon = (X == 0).mean(axis=0) ** 4 + 1e-8
        self.mu = Xs.mean(axis=0)
        self.sigma = Xs.std(axis=0) + self.epsilon
        return self

    def transform(self, X):
        Xs = np.sqrt(np.clip(X, 0, None))
        return ((Xs - self.mu) * (Xs != 0)) / self.sigma

    @classmethod
    def from_params(cls, mu, sigma):
        obj = cls()
        obj.mu = np.asarray(mu)
        obj.sigma = np.asarray(sigma)
        obj.epsilon = None
        return obj
