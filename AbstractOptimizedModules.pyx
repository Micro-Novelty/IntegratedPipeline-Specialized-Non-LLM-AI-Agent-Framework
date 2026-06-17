# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=False

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, tanh, fabs, log1p, sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


# ── activation functions ──────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_sigmoid(np.ndarray[DTYPE_t, ndim=1] x):
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=DTYPE)
    cdef Py_ssize_t i
    cdef DTYPE_t xi
    for i in range(n):
        xi = x[i]
        if xi > 500.0:        # clip guard in C, no numpy clip needed
            out[i] = 1.0
        elif xi < -500.0:
            out[i] = 0.0
        else:
            out[i] = 1.0 / (1.0 + exp(-xi))
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_sigmoid_deriv(np.ndarray[DTYPE_t, ndim=1] s):
    cdef Py_ssize_t n = s.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=DTYPE)
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = s[i] * (1.0 - s[i])
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_tanh_deriv(np.ndarray[DTYPE_t, ndim=1] t):
    cdef Py_ssize_t n = t.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=DTYPE)
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = 1.0 - t[i] * t[i]
    return out


# ── LSTM cell forward ─────────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_lstm_cell_forward(
    np.ndarray[DTYPE_t, ndim=2] x_seq,
    np.ndarray[DTYPE_t, ndim=2] W,
    np.ndarray[DTYPE_t, ndim=1] b,
    Py_ssize_t input_size,
    Py_ssize_t hidden_size
):
    cdef Py_ssize_t T      = x_seq.shape[0]
    cdef Py_ssize_t H      = hidden_size
    cdef Py_ssize_t H2     = H * 2
    cdef Py_ssize_t H3     = H * 3
    cdef Py_ssize_t xh_len = input_size + H
    cdef Py_ssize_t t, j, k

    cdef np.ndarray[DTYPE_t, ndim=1] h      = np.zeros(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] c      = np.zeros(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] xh     = np.empty(xh_len, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] z      = np.empty(4 * H,  dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] f      = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] i_g    = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] g_g    = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] o_g    = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] c_new  = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] tanh_c = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] h_new  = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] hs     = np.zeros((T, H), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] cs     = np.zeros((T, H), dtype=DTYPE)

    # W dimensions for pure C matmul
    cdef Py_ssize_t W_rows = W.shape[0]  # 4H
    cdef Py_ssize_t W_cols = W.shape[1]  # input_size + H
    cdef DTYPE_t acc

    cache = []

    for t in range(T):
        x = x_seq[t]
        if x.shape[0] < input_size:
            x = np.pad(x, (0, input_size - x.shape[0]))
        elif x.shape[0] > input_size:
            x = x[:input_size]

        xh[:input_size] = x
        xh[input_size:] = h

        # pure C matmul: z = W @ xh + b
        for j in range(W_rows):
            acc = b[j]
            for k in range(W_cols):
                acc += W[j, k] * xh[k]
            z[j] = acc

        # gate activations
        for j in range(H):
            f[j]      = 1.0 / (1.0 + exp(-z[j]))
            i_g[j]    = 1.0 / (1.0 + exp(-z[H  + j]))
            g_g[j]    = tanh(z[H2 + j])
            o_g[j]    = 1.0 / (1.0 + exp(-z[H3 + j]))
            c_new[j]  = f[j] * c[j] + i_g[j] * g_g[j]
            tanh_c[j] = tanh(c_new[j])
            h_new[j]  = o_g[j] * tanh_c[j]

        cache.append((
            x.copy(), h.copy(), c.copy(),
            f.copy(), i_g.copy(), g_g.copy(),
            o_g.copy(), c_new.copy(), tanh_c.copy(),
            xh.copy()
        ))

        # swap buffers instead of copy
        h = h_new
        c = c_new
        h_new  = np.empty(H, dtype=DTYPE)
        c_new  = np.empty(H, dtype=DTYPE)

        hs[t] = h
        cs[t] = c

    return hs, cs, cache


# ── multi head attention ──────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_project_heads(
    np.ndarray[DTYPE_t, ndim=3] x,
    np.ndarray[DTYPE_t, ndim=3] W,
    Py_ssize_t B,
    Py_ssize_t S,
    Py_ssize_t num_heads,
    Py_ssize_t D,
    Py_ssize_t M
):
    cdef np.ndarray[DTYPE_t, ndim=4] out = np.zeros(
        (B, num_heads, S, M), dtype=DTYPE
    )
    cdef Py_ssize_t b, h, s, d, m
    cdef DTYPE_t acc

    for b in range(B):
        for h in range(num_heads):
            for s in range(S):
                for m in range(M):
                    acc = 0.0
                    for d in range(D):
                        acc += x[b, s, d] * W[h, d, m]
                    out[b, h, s, m] = acc
    return out


# ── AME encoder ───────────────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_ame_encoder(np.ndarray[DTYPE_t, ndim=2] x):
    cdef Py_ssize_t T = x.shape[0]
    cdef Py_ssize_t D = x.shape[1]
    cdef DTYPE_t grad_energy = 0.0
    cdef DTYPE_t x_mag       = 0.0
    cdef DTYPE_t norm, diff
    cdef Py_ssize_t t, d

    for t in range(T):
        norm = 0.0
        for d in range(D):
            norm += x[t, d] * x[t, d]
        x_mag += sqrt(norm)      

    x_mag /= T

    for t in range(1, T):
        norm = 0.0
        for d in range(D):
            diff  = x[t, d] - x[t-1, d]
            norm += diff * diff
        grad_energy += sqrt(norm)  # libc sqrt

    if T > 1:
        grad_energy /= (T - 1)

    return log1p(x_mag) * log1p(grad_energy)


# ── Anisotropy ────────────────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_anisotropy(np.ndarray[DTYPE_t, ndim=2] x):
    """
    Welford online mean+std of finite difference norms.
    Eliminates Python list comprehension over np.gradient output.
    """
    cdef Py_ssize_t T      = x.shape[0]
    cdef Py_ssize_t D      = x.shape[1]
    cdef Py_ssize_t t, d
    cdef DTYPE_t norm, diff
    cdef DTYPE_t mean_norm = 0.0
    cdef DTYPE_t M2        = 0.0
    cdef DTYPE_t delta, delta2
    cdef Py_ssize_t count  = 0
    cdef DTYPE_t eps       = 1e-5

    for t in range(1, T):
        norm = 0.0
        for d in range(D):
            diff  = x[t, d] - x[t-1, d]
            norm += diff * diff
        norm = sqrt(norm)

        count += 1
        delta  = norm - mean_norm
        mean_norm += delta / count
        delta2 = norm - mean_norm
        M2    += delta * delta2

    if count < 2:
        return eps

    cdef DTYPE_t std_norm = sqrt(M2 / (count - 1))
    return std_norm / (mean_norm + eps)


# ── Cosine similarity ─────────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_cosine_similarity(
    np.ndarray[DTYPE_t, ndim=1] a,
    np.ndarray[DTYPE_t, ndim=1] b
):
    cdef Py_ssize_t n = min(a.shape[0], b.shape[0])
    cdef Py_ssize_t i
    cdef DTYPE_t dot    = 0.0
    cdef DTYPE_t norm_a = 0.0
    cdef DTYPE_t norm_b = 0.0

    for i in range(n):
        dot    += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]

    return dot / (sqrt(norm_a) * sqrt(norm_b) + 1e-8)


# ── Softmax 2D ────────────────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_softmax_2d(np.ndarray[DTYPE_t, ndim=2] x):
    cdef Py_ssize_t B = x.shape[0]
    cdef Py_ssize_t C = x.shape[1]
    cdef Py_ssize_t b, c
    cdef DTYPE_t max_val, sum_val
    cdef np.ndarray[DTYPE_t, ndim=2] out = np.empty((B, C), dtype=DTYPE)

    for b in range(B):
        max_val = x[b, 0]
        for c in range(1, C):
            if x[b, c] > max_val:
                max_val = x[b, c]

        sum_val = 0.0
        for c in range(C):
            out[b, c] = exp(x[b, c] - max_val)
            sum_val  += out[b, c]

        for c in range(C):
            out[b, c] /= sum_val + 1e-8

    return out
