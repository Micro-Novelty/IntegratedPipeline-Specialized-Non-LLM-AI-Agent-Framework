# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: initializedcheck=True
# cython: cdivision=False

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, tanh, fabs, log1p, sqrt, log

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# _________ replaces einsum equations directly with Cython for Dynamic backward _________
@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_qkv_weight_grad(
    np.ndarray[DTYPE_t, ndim=3] x,        # (B, S, D)
    np.ndarray[DTYPE_t, ndim=4] d_proj,   # (B, H, S, M) — d_Q, d_K, or d_V
    Py_ssize_t B, Py_ssize_t S, Py_ssize_t H,
    Py_ssize_t D, Py_ssize_t M
):
    """
    Replaces einsum('bsd,bhsm->hdm', x, d_proj).
    Computes weight gradient: for each head h, dim d, dim m:
      sum over b,s of x[b,s,d] * d_proj[b,h,s,m]
    """
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros((H, D, M), dtype=DTYPE)
    cdef Py_ssize_t b, h, s, d, m
    cdef DTYPE_t acc

    for h in range(H):
        for d in range(D):
            for m in range(M):
                acc = 0.0
                for b in range(B):
                    for s in range(S):
                        acc += x[b, s, d] * d_proj[b, h, s, m]
                out[h, d, m] = acc
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_qkv_input_grad(
    np.ndarray[DTYPE_t, ndim=4] d_proj,   # (B, H, S, M)
    np.ndarray[DTYPE_t, ndim=3] W,        # (H, D, M)
    Py_ssize_t B, Py_ssize_t S, Py_ssize_t H,
    Py_ssize_t D, Py_ssize_t M
):
    """
    Replaces einsum('bhsm,hdm->bsd', d_proj, W).
    Computes input gradient: for each batch b, position s, dim d:
      sum over h,m of d_proj[b,h,s,m] * W[h,d,m]
    """
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros((B, S, D), dtype=DTYPE)
    cdef Py_ssize_t b, h, s, d, m
    cdef DTYPE_t acc

    for b in range(B):
        for s in range(S):
            for d in range(D):
                acc = 0.0
                for h in range(H):
                    for m in range(M):
                        acc += d_proj[b, h, s, m] * W[h, d, m]
                out[b, s, d] = acc
    return out

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


# _______ dynamic ensemble method __________

@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_dynamic_weighted_ensemble(
    np.ndarray[DTYPE_t, ndim=2] trans_probs,    # (B, n_trans)
    np.ndarray[DTYPE_t, ndim=2] mlp_probs,      # (B, n_mlp)
    np.ndarray[DTYPE_t, ndim=2] attn_flat,      # (B, attn_features) — pre-flattened
    np.ndarray[DTYPE_t, ndim=2] lstm_probs,     # (B, n_lstm) or None→zeros(B,1)
    np.ndarray[DTYPE_t, ndim=1] lstm_weight_hints,  # (B,) per-sample hint
    DTYPE_t confidence_threshold,
    bint has_lstm                                # True if lstm_probs is here
):
    cdef Py_ssize_t B          = trans_probs.shape[0]
    cdef Py_ssize_t n_trans    = trans_probs.shape[1]
    cdef Py_ssize_t n_mlp      = mlp_probs.shape[1]
    cdef Py_ssize_t n_lstm     = lstm_probs.shape[1] if has_lstm else 0
    cdef Py_ssize_t n_classes  = n_trans
    if n_mlp  > n_classes: n_classes = n_mlp
    if n_lstm > n_classes: n_classes = n_lstm

    cdef np.ndarray[DTYPE_t, ndim=2] ensemble = np.zeros((B, n_classes), dtype=DTYPE)

    cdef Py_ssize_t i, j
    cdef DTYPE_t trans_sum, mlp_sum, lstm_sum
    cdef DTYPE_t trans_pred_val, mlp_pred_val, lstm_pred_val
    cdef Py_ssize_t trans_pred, mlp_pred, lstm_pred
    cdef DTYPE_t agreement, lstm_agreement
    cdef DTYPE_t attn_focus, attn_growth, anisotropy, attn_limit
    cdef DTYPE_t attn_mean, attn_std, attn_sq_sum
    cdef DTYPE_t mlp_entropy, mlp_conf_factor
    cdef DTYPE_t trans_conf_factor, trans_weight, mlp_weight, lstm_weight, total
    cdef Py_ssize_t attn_len = attn_flat.shape[1]

    # preallocate row buffers — reused each iteration
    cdef np.ndarray[DTYPE_t, ndim=1] trans_row = np.empty(n_classes, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] mlp_row   = np.empty(n_classes, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] lstm_row  = np.empty(n_classes, dtype=DTYPE)

    for i in range(B):

        # ── build aligned rows ───────────────────────────────────────
        for j in range(n_classes):
            trans_row[j] = trans_probs[i, j] if j < n_trans else 0.0
            mlp_row[j]   = mlp_probs[i, j]   if j < n_mlp  else 0.0
            lstm_row[j]  = lstm_probs[i, j]   if (has_lstm and j < n_lstm) else 0.0

        # normalize rows
        trans_sum = 0.0
        mlp_sum   = 0.0
        lstm_sum  = 0.0
        for j in range(n_classes):
            trans_sum += trans_row[j]
            mlp_sum   += mlp_row[j]
            lstm_sum  += lstm_row[j]
        trans_sum += 1e-8
        mlp_sum   += 1e-8
        lstm_sum  += 1e-8
        for j in range(n_classes):
            trans_row[j] /= trans_sum
            mlp_row[j]   /= mlp_sum
            if has_lstm:
                lstm_row[j] /= lstm_sum

        # ── argmax predictions ───────────────────────────────────────
        trans_pred = 0
        mlp_pred   = 0
        lstm_pred  = 0
        trans_pred_val = trans_probs[i, 0]
        mlp_pred_val   = mlp_probs[i, 0]
        for j in range(1, n_trans):
            if trans_probs[i, j] > trans_pred_val:
                trans_pred_val = trans_probs[i, j]
                trans_pred = j
        for j in range(1, n_mlp):
            if mlp_probs[i, j] > mlp_pred_val:
                mlp_pred_val = mlp_probs[i, j]
                mlp_pred = j
        if has_lstm:
            lstm_pred_val = lstm_probs[i, 0]
            for j in range(1, n_lstm):
                if lstm_probs[i, j] > lstm_pred_val:
                    lstm_pred_val = lstm_probs[i, j]
                    lstm_pred = j

        agreement = 1.0 if trans_pred == mlp_pred else 0.3

        # ── attention confidence factor ───────────────────────────────
        # attn_flat[i] is the pre-flattened attention row
        # compute mean and std in one C pass (Welford)
        attn_mean   = 0.0
        attn_sq_sum = 0.0
        for j in range(attn_len):
            attn_mean += attn_flat[i, j]
        attn_mean /= (attn_len + 1e-8)
        for j in range(attn_len):
            attn_sq_sum += (attn_flat[i, j] - attn_mean) ** 2
        attn_focus  = (attn_sq_sum / (attn_len + 1e-8)) ** 0.5
        attn_growth = 1.0 / (1.0 + exp(-attn_focus))

        # anisotropy approximation — ratio of std to mean
        anisotropy      = attn_focus / (attn_mean + 1e-8)
        attn_limit      = (1.0 - attn_focus + attn_growth) * anisotropy
        trans_conf_factor = attn_growth + attn_limit * attn_focus

        # ── MLP entropy confidence ────────────────────────────────────
        mlp_entropy = 0.0
        for j in range(n_mlp):
            if mlp_probs[i, j] > 1e-8:
                mlp_entropy -= mlp_probs[i, j] * log(mlp_probs[i, j] + 1e-8)
        mlp_conf_factor = 1.0 / (1.0 + mlp_entropy)

        # ── weights ───────────────────────────────────────────────────
        trans_weight = trans_conf_factor * (1.0 + agreement) / 2.0
        mlp_weight   = mlp_conf_factor   * (1.0 + agreement) / 2.0

        if has_lstm:
            lstm_agreement = 1.0 if (lstm_pred == trans_pred or
                                     lstm_pred == mlp_pred) \
                             else confidence_threshold
            lstm_weight = lstm_weight_hints[i] * (1.0 + lstm_agreement) / 2.0
            total = trans_weight + mlp_weight + lstm_weight + 1e-8
            trans_weight /= total
            mlp_weight   /= total
            lstm_weight  /= total
            for j in range(n_classes):
                ensemble[i, j] = (trans_weight * trans_row[j] +
                                  mlp_weight   * mlp_row[j]   +
                                  lstm_weight  * lstm_row[j])
        else:
            total = trans_weight + mlp_weight + 1e-8
            trans_weight /= total
            mlp_weight   /= total
            for j in range(n_classes):
                ensemble[i, j] = (trans_weight * trans_row[j] +
                                  mlp_weight   * mlp_row[j])

    return ensemble

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
