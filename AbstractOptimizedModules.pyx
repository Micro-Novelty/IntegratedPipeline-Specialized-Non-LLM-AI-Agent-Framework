
# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=False

# Optimized Abstract Modules implemented in Cython.
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, tanh, fabs
from libc.math cimport log1p

# ── type aliases ──────────────────────────────
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# ── activation functions ──────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_sigmoid(np.ndarray[DTYPE_t, ndim=1] x):
    """Vectorized sigmoid — avoids numpy function call overhead."""
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=DTYPE)
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = 1.0 / (1.0 + exp(-x[i]))
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_sigmoid_deriv(np.ndarray[DTYPE_t, ndim=1] s):
    """s is already sigmoid output."""
    cdef Py_ssize_t n = s.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=DTYPE)
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = s[i] * (1.0 - s[i])
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_tanh_deriv(np.ndarray[DTYPE_t, ndim=1] t):
    """t is already tanh output."""
    cdef Py_ssize_t  n = t.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=DTYPE)
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = 1.0 - t[i] * t[i]
    return out

# ── LSTM cell forward ─────────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_lstm_cell_forward(
    np.ndarray[DTYPE_t, ndim=2] x_seq,   # (T, input_size)
    np.ndarray[DTYPE_t, ndim=2] W,       # (4H, input+H)
    np.ndarray[DTYPE_t, ndim=1] b,       # (4H,)
    Py_ssize_t input_size,
    Py_ssize_t hidden_size
):
    cdef Py_ssize_t T   = x_seq.shape[0]
    cdef Py_ssize_t H      = hidden_size
    cdef Py_ssize_t H2     = H * 2
    cdef Py_ssize_t H3     = H * 3
    cdef Py_ssize_t xh_len = input_size + H
    cdef Py_ssize_t t, j

    cdef np.ndarray[DTYPE_t, ndim=1] h     = np.zeros(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] c     = np.zeros(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] xh    = np.empty(xh_len, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] z     = np.empty(4 * H,  dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] f     = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] i_g   = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] g_g   = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] o_g   = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] c_new = np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] tanh_c= np.empty(H,      dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] h_new = np.empty(H,      dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=2] hs = np.zeros((T, H), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] cs = np.zeros((T, H), dtype=DTYPE)

    cache = []
    cdef double val

    for t in range(T):
        # fill xh buffer
        x = x_seq[t]
        if x.shape[0] < input_size:
            x = np.pad(x, (0, input_size - x.shape[0]))
        elif x.shape[0] > input_size:
            x = x[:input_size]

        xh[:input_size] = x
        xh[input_size:] = h

        # z = W @ xh + b
        z = W @ xh + b

        # gate activations using libc math — no numpy call overhead
        for j in range(H):
            f[j]    = 1.0 / (1.0 + exp(-z[j]))
            i_g[j]  = 1.0 / (1.0 + exp(-z[H  + j]))
            g_g[j]  = tanh(z[H2 + j])
            o_g[j]  = 1.0 / (1.0 + exp(-z[H3 + j]))
            c_new[j]  = f[j] * c[j] + i_g[j] * g_g[j]
            tanh_c[j] = tanh(c_new[j])
            h_new[j]  = o_g[j] * tanh_c[j]

        cache.append((
            x.copy(), h.copy(), c.copy(),
            f.copy(), i_g.copy(), g_g.copy(),
            o_g.copy(), c_new.copy(), tanh_c.copy(),
            xh.copy()
        ))

        h = h_new.copy()
        c = c_new.copy()
        hs[t] = h
        cs[t] = c

    return hs, cs, cache

# ── multi head attention ──────────────────────
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_project_heads(
    np.ndarray[DTYPE_t, ndim=3] x,       # (B, S, D)
    np.ndarray[DTYPE_t, ndim=3] W,       # (H, D, M)
    Py_ssize_t B, Py_ssize_t S, int num_heads,
    Py_ssize_t D, Py_ssize_t M
):
    """Replace einsum('bsd,hdm->bhsm') with direct C loop."""
    cdef np.ndarray[DTYPE_t, ndim=4] out = np.zeros(
        (B, num_heads, S, M), dtype=DTYPE
    )
    cdef Py_ssize_t b, h, s, d, m
    cdef double acc

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
    """
    Faster AME — avoids np.gradient Python overhead.
    x: (T, D)
    """
    cdef Py_ssize_t T = x.shape[0]
    cdef Py_ssize_t D = x.shape[1]
    cdef double grad_energy = 0.0
    cdef double x_mag       = 0.0
    cdef double norm, diff
    cdef Py_ssize_t t, d

    for t in range(T):
        norm = 0.0
        for d in range(D):
            norm += x[t, d] * x[t, d]
        x_mag += norm ** 0.5

    x_mag /= T

    # gradient approximation via finite difference
    for t in range(1, T):
        norm = 0.0
        for d in range(D):
            diff  = x[t, d] - x[t-1, d]
            norm += diff * diff
        grad_energy += norm ** 0.5

    if T > 1:
        grad_energy /= (T - 1)

    # log1p(x_mag) * log1p(grad_energy)
    cdef double AME
    AME = (
        (x_mag      + 1.0) if x_mag      > 0 else 1.0
    )
    AME = AME * ((grad_energy + 1.0) if grad_energy > 0 else 1.0)

    return log1p(x_mag) * log1p(grad_energy)