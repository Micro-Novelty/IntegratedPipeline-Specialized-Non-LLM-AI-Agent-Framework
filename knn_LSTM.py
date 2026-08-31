import numpy as np
import random
import math
import heapq
from typing import Any, Callable, Tuple, Optional, Dict, List

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(s):          # s = sigmoid(x) already computed
    return s * (1.0 - s)

def tanh_deriv(t):             # t = tanh(x) already computed
    return 1.0 - t ** 2


class GeometricWeightShaping:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    

    def eigenvalue_encoder(self, x):
        eps = 1e-5
        raw_X = np.asarray(x)
        AME = self.AME_Encoder(raw_X)  
        AMR = 1.0 / (1.0 + np.exp(-AME)) + eps
        mag = np.mean(np.linalg.norm(raw_X, axis=-1))

        if raw_X.ndim > 2:
            raw_X = raw_X.reshape(raw_X.shape[0], -1)

        anisotropy = self.anisotropy_measurement(raw_X)

        structured_noise = np.random.uniform(0, mag, size=raw_X.shape)
        X = np.vstack((raw_X, structured_noise))
        if X.ndim == 2 and X.shape[1] == 1:
            X = np.hstack((raw_X, structured_noise))

        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]

        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        energy_sigmoid_growth = 1.0 / (1.0 + np.exp(-energy))
        energy_consistency = np.std(energy_sigmoid_growth)
        k = np.searchsorted(energy, 0.90) + 1     # +1 converts 0-based index to count

        trA = k / (1.0 - anisotropy) + eps  
        trB = (1/2 + energy_consistency) / (1.0 + trA**2)
        trC = (1/6 + AMR) / (1.0 - trB**2) + eps

        if np.isnan(trC) or np.isinf(trC):
            trC = anisotropy * (trB**2 - 1.0) + eps
            if np.isnan(trC) or np.isinf(trC):
                trC = (1.0 - AMR)

        min_val = min(trC, 0) 
        max_val = max(trC, 0) 
        floating_point = np.random.uniform(min_val, max_val, size=X.shape) 
        return k, floating_point, structured_noise


    def spectral_signature(self, x, structured_noise, k=5):
        raw_X = np.asarray(x, dtype=np.float64)

        if raw_X.ndim > 2:
            X = raw_X.reshape(raw_X.shape[0], -1)
        else:
            X = raw_X.reshape(raw_X.shape[0], -1)

        X = np.atleast_2d(X)

        if X.ndim == 2 and X.shape[1] == 1:
            # normalize structured_noise to 2D matching X's row count
            noise = np.asarray(structured_noise, dtype=np.float64)

            if noise.ndim == 1:
                # reshape to (n_samples, n_noise_features)
                # if noise length matches X's row count, treat as column vector
                if noise.shape[0] == X.shape[0]:
                    noise = noise.reshape(-1, 1)
                else:
                    # noise is a flat feature vector not aligned to X's rows —
                    # broadcast it across all rows instead of stacking blindly
                    noise = np.tile(noise.reshape(1, -1), (X.shape[0], 1))
            elif noise.ndim > 2:
                noise = noise.reshape(noise.shape[0], -1)

            # align row counts before hstack
            if noise.shape[0] != X.shape[0]:
                min_rows = min(noise.shape[0], X.shape[0])
                X     = X[:min_rows]
                noise = noise[:min_rows]
                print(f'[⚠️] spectral_signature: row count mismatch, '
                    f'truncated to {min_rows} rows')

            X = np.hstack((X, noise))

        # guard against degenerate covariance — need at least 2 samples
        if X.shape[0] < 2:
            print(f'[⚠️] spectral_signature: only {X.shape[0]} sample(s), '
                f'cannot compute covariance — returning zeros')
            return np.zeros(k)

        try:
            cov     = np.cov(X, rowvar=False, ddof=1)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eig_sum = eigvals.sum()
            if eig_sum <= 1e-8:
                return np.zeros(min(k, len(eigvals)))
            return eigvals[:k] / (eig_sum + 1e-8)
        except np.linalg.LinAlgError as e:
            print(f'[⚠️] spectral_signature: eigendecomposition failed: {e}')
            return np.zeros(k)


    def spectral_similarity(self, a, b, structured_noise):
        sa = self.spectral_signature(a, structured_noise)
        sb = self.spectral_signature(b, structured_noise)
        if sa.shape != sb.shape:
            min_rows = min(sa.shape[0], sb.shape[0])

            sa = sa[:min_rows]
            sb = sb[:min_rows]

        return np.exp(-np.linalg.norm(sa - sb))

    # abstract modelling error provides the model how to better process weights when the data complexity has little geometric complexity
    def AME_Encoder(self, x):
        X = np.asarray(x)

        if len(X) == 0:
            print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
            return 0.0

        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        mean_vector_mag  = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))
        # Regular AME Equations, higher AME provides capabilities for the model to experience errors during abstraction
        # Lower AME means lower chance for un optimal abstraction.

        AME =  np.log1p(X_mag) * np.log1p(mean_vector_mag) 
        return AME

    # anisotropy provides the model the standard complexity of the data geometry, allowing it to know how complex the data needs to be processed.
    def anisotropy_measurement(self, x):
        eps = 1e-5
    
        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps

        return anisotropy

    # weight shaping provides directional context in which how the data should be processed in order to align with the data geometry
    def abstract_weight_shaping(self, x):
        input_size = self.input_size
        output_size = self.output_size


        eps = 1e-5
        x = np.asarray(x)

        rng = np.random.default_rng()

        anisotropy = self.anisotropy_measurement(x)
        mag = np.mean(np.linalg.norm(x))

        k, floating_point, structured_noise = self.eigenvalue_encoder(x)
        AME = self.AME_Encoder(x)
        AMR = 1.0 / (1.0 + np.exp(-AME)) # abstract modelling rate        

        spectral_similarity = self.spectral_similarity(x, floating_point, structured_noise)

        AEL = (0.3 + spectral_similarity + eps) * anisotropy 

        scaled_anisotropy = anisotropy / (anisotropy + 1.0)
        
        abstraction_efficiency = (1.0 + AEL) * (1.0 - AMR) + eps
        abstraction_efficiency = (k + AEL) * (1.0 - AMR) + eps

        if np.isnan(abstraction_efficiency) or np.isinf(abstraction_efficiency):
            abstraction_efficiency = (1 - AMR) + eps

        abstract_context = rng.uniform(0, abstraction_efficiency, size=(input_size, output_size)) 

        return abstract_context



    def weight_shaping(self, x, type=None):
        if np.isnan(x).any() or np.isinf(x).any():
            x = np.nan_to_num(x, nan=0.0, posinf=1e99, neginf=-1e99)     

        if isinstance(x, list):
            x = np.asarray(x)

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if np.std(x) == 0:
            x = np.random.uniform(0, 1, size=x.shape)

        abstract_context = self.abstract_weight_shaping(x)
        abstract_context /= np.max(np.abs(abstract_context)) + 1e-5  # Normalized to [-1, 1]

        return abstract_context



class HNSW:
    def __init__(self, dim, M=24, ef_construction=264, metric='euclidean', seed=None):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.metric = metric

        self.mL = 1.0 / math.log(M)
        self.vectors: List[np.ndarray] = []
        self.neighbors: List[List[set]] = []
        self.max_level_of: List[int] = []

        self.entry_point = -1
        self.max_level = -1
        self.seed = seed
        self._rng = random.Random(seed)

    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return float(np.linalg.norm(a - b))
        elif self.metric == 'cosine':
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        else:
            raise ValueError(f'Unknown Metric: {self.metric}')

    def _random_level(self):
        return int(-math.log(self._rng.random()) * self.mL)

    def _search_layer(self, query, entry_points, ef, layer):
        visited = set(entry_points)

        candidates = [(self._distance(query, self.vectors[ep]), ep) for ep in entry_points]
        heapq.heapify(candidates)

        found = [(-d, n) for d, n in candidates]
        heapq.heapify(found)

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst_found = -found[0][0]
 
            if dist_c > worst_found and len(found) >= ef:
                break

            for neighbor in self.neighbors[c][layer]:
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                d = self._distance(query, self.vectors[neighbor])
                worst_found = -found[0][0]

                if d < worst_found or len(found) < ef:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(found, (-d, neighbor))
                    if len(found) > ef:
                        heapq.heappop(found)

        return [(-d, n) for d, n in found]

    def _select_neighbors_heuristic(self, query, candidates, m):
        candidates = sorted(candidates, key=lambda x: x[0])
        selected: List[Tuple[float, int]] = []

        for dist_to_query, c in candidates:
            if len(selected) >= m:
                break

            vec_c = self.vectors[c]
            competitive = True
            for _, s in selected:
                if self._distance(vec_c, self.vectors[s]) < dist_to_query:
                    competitive = False
                    break
            if competitive:
                selected.append((dist_to_query, c))

        return [n for _, n in selected]

    def insert(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        node_id = len(self.vectors)
        node_level = self._random_level()

        self.vectors.append(vector)
        self.neighbors.append([set() for _ in range(node_level + 1)])
        self.max_level_of.append(node_level)

        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_level = node_level
            return node_id

        ep = [self.entry_point]
        for layer in range(self.max_level, node_level, -1):
            result = self._search_layer(vector, ep, ef=1, layer=layer)
            ep = [min(result, key=lambda x: x[0])[1]]

        
        for layer in range(min(node_level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, ep, ef=self.ef_construction, layer=layer)
            m = self.M0 if layer == 0 else self.M
            chosen = self._select_neighbors_heuristic(vector, candidates, m)

            self.neighbors[node_id][layer] = set(chosen)
            for neighbor in chosen:
                self.neighbors[neighbor][layer].add(node_id)
                self._prune(neighbor, layer)

            ep = [n for _, n in candidates]
        
        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = node_id

        return node_id

    
    def _prune(self, node, layer):
        m = self.M0 if layer == 0 else self.M
        neighs = self.neighbors[node][layer]
        if len(neighs) <= m:
            return

        vec = self.vectors[node]
        scored = [(self._distance(vec, self.vectors[n]), n) for n in neighs]
        kept = self._select_neighbors_heuristic(vec, scored, m)
        self.neighbors[node][layer] = set(kept)

    def search(self, query, k, ef):
        if self.entry_point == -1:
            return []
        query = np.asarray(query, dtype=np.float32)
        ef = ef or max(k, self.ef_construction)
        ep = [self.entry_point]

        for layer in range(self.max_level, 0, -1):
            result = self._search_layer(query, ep, ef=1, layer=layer)
            ep = [min(result, key=lambda x: x[0])[1]]
        
        candidates = self._search_layer(query, ep, ef=1, layer=0)
        candidates.sort(key=lambda x: x[0])
        return [(n, d) for d, n in candidates[:k]]



class PerCellMemory:
    def __init__(self, inp_size, dim_size, capacity=8000, k=8, metric='euclidean', evict_fraction=0.1):
        self.inp_size = inp_size
        self.dim_size = dim_size
        self.k = k
        self.dim_ratio = self.dim_size // self.k
        self.memory_capacity = capacity

        self.evict_batch = max(1, int(capacity * evict_fraction))
        self.indices = [HNSW(dim=self.dim_size, M=16, ef_construction=100, metric='euclidean') for _ in range(self.dim_ratio)]

        self._values_1 = [[] for _ in range(self.dim_size)]
        self._values_2 = [[] for _ in range(self.dim_size)]
        self._values_3 = [[] for _ in range(self.dim_size)]
        self._count = [0] * self.dim_size

    def write(self, c, i, o, g):
        B = c.shape[0]
        for hid in range(B):
            if self._count[hid] >= self.memory_capacity:
                continue

            if hid < len(self.indices):
                self.indices[hid].insert(c[hid])
                self._count[hid] += 1
            
            if self._count[hid] > self.memory_capacity:
                self._evict_and_rebuild(hid)
    


    def _evict_and_rebuild(self, h):
        keep_from = self.evict_batch
        surviving_vectors = self.indices[h].vectors[keep_from:]
        surviving_values_1 = self._values_1[h][keep_from:]
        surviving_values_2 = self._values_2[h][keep_from:]
        surviving_values_3 = self._values_3[h][keep_from]

        fresh_index = HNSW(dim=self.head_dim, M=self.indices[h].M,
                            ef_construction=self.indices[h].ef_construction,
                            metric=self.metric)
        for vec in surviving_vectors:
            fresh_index.insert(vec)
 
        self.indices[h] = fresh_index
        self._count[h] = len(surviving_values_1) + len(surviving_values_2) + len(surviving_values_3) // 3


    def retrieve(self, f):
        B = f.shape[0]

        k = self.k
        c_memory = np.zeros((B, k), dtype=np.float32)
        i_memory = np.zeros((B, k), dtype=np.float32)
        o_memory = np.zeros((B, k), dtype=np.float32)
        g_memory = np.zeros((B, k), dtype=np.float32)
        mem_mask = np.zeros((B, k), dtype=np.float32)

        for t in range(B):
            if self._count[t] == 0:
                continue
            results = self.indices[t].search(f[t], k=k, ef=max(k, 50))
            for j, (node_id, _dist) in enumerate(results):
                c_memory[t, j] = self.indices[t].vectors[node_id]

                if node_id < len(self._values_1[t]):
                    i_memory[t, j] = self._values_1[t][node_id]
                if node_id < len(self._values_2[t]):
                    o_memory[t, j] = self._values_2[t][node_id]
                if node_id < len(self._values_3[t]):
                    g_memory[t, j] = self._values_3[t][node_id]
                mem_mask[t, j] = 1.0

        return c_memory, i_memory, o_memory, g_memory, mem_mask

        


class LSTMCell:
    """
    Single LSTM cell operating on one time-step.

    Gate layout (all concatenated into one weight matrix for speed):
        W shape: (4*hidden, input + hidden)
        b shape: (4*hidden,)

    Slice order: [forget | input | gate (candidate) | output]
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        np.random.seed(seed)

        # Xavier / Glorot init
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * scale
        self.b = np.zeros((4 * hidden_size,))
        self.b[:hidden_size] = 1.0  # forget gate bias init to 1.0

        # Output projection: hidden → output
        self.Wy = np.random.randn(hidden_size, hidden_size) * scale
        self.by = np.zeros((hidden_size,))

    # ── slicing helpers utility functions ──────────────────────
    def _f(self, v): return v[:self.hidden_size]
    def _i(self, v): return v[self.hidden_size:2*self.hidden_size]
    def _g(self, v): return v[2*self.hidden_size:3*self.hidden_size]
    def _o(self, v): return v[3*self.hidden_size:]


    # _________ forward method for cell class _____________
    def forward(self, x_seq: np.ndarray, h0=None, c0=None):
        # Optimized LSTM Cell Implementation with Cython based Language.

        T              = x_seq.shape[0]
        H              = self.hidden_size
        expected_input = self.input_size

        h = np.zeros(H) if h0 is None else h0.copy()
        c = np.zeros(H) if c0 is None else c0.copy()

        hs    = np.zeros((T, H))
        cs    = np.zeros((T, H))
        cache = []

        # preallocate xh buffer once only here.
        xh = np.empty(expected_input + H)

        for t in range(T):
            x = x_seq[t]
            if x.ndim == 0:
                x = x.reshape(1)
            if x.shape[0] < expected_input:
                x = np.pad(x, (0, expected_input - x.shape[0]))
            elif x.shape[0] > expected_input:
                x = x[:expected_input]

            # write into buffer, no allocation
            xh[:expected_input] = x
            xh[expected_input:] = h

            z      = self.W @ xh + self.b
            H1, H2, H3 = H, H * 2, H * 3

            # direct slices, no method calls
            f      = sigmoid(z[:H1])
            i      = sigmoid(z[H1:H2])
            g      = np.tanh(z[H2:H3])
            o      = sigmoid(z[H3:])

            c_new  = f * c + i * g
            c_new = np.clip(c_new, -10.0, 10.0)  # prevent overflow in tanh

            tanh_c = np.tanh(c_new)
            h_new  = o * tanh_c

            # store copies — h/c will be overwritten next iteration 
            cache.append((x.copy(), h.copy(), c.copy(),
                        f, i, g, o, c_new, tanh_c, xh.copy()))
            h, c = h_new, c_new
            hs[t] = h
            cs[t] = c


        return hs, cs, cache

    # ________ backward method for Cell class __________
    def backward(self, dhs: np.ndarray, cache,
                dh_next=None, dc_next=None, T_limit=None):
        # T_limit avoids slicing cache list externally Later.
        T = T_limit if T_limit is not None else len(cache)
        H = self.hidden_size

        dW     = np.zeros_like(self.W)
        db     = np.zeros_like(self.b)
        dh     = np.zeros(H) if dh_next is None else np.ascontiguousarray(dh_next.copy())
        dc     = np.zeros(H) if dc_next is None else np.ascontiguousarray(dc_next.copy())
        dx_seq = np.zeros((T, self.input_size))
        if _OPT_AVAILABLE:
            grads, dx_seq, dh, dc = optimized_lstm_cell_backward(
                np.ascontiguousarray(dhs, dtype=np.float64),
                cache,
                np.ascontiguousarray(self.W, dtype=np.float64),
                self.input_size,
                self.hidden_size,
                dh,
                dc,
                T
            )
            return grads, dx_seq, dh, dc

        # preallocate dz buffer once
        dz     = np.empty(4 * H)
        H1, H2, H3 = H, H * 2, H * 3

        for t in reversed(range(T)):
            x, h_prev, c_prev, f, i, g, o, c_new, tanh_c, xh = cache[t]

            dh_total = dhs[t] + dh

            do     = dh_total * tanh_c
            dtanhc = dh_total * o
            dc_new = dtanhc * tanh_deriv(tanh_c) + dc

            df = dc_new * c_prev
            di = dc_new * g
            dg = dc_new * i
            dc = dc_new * f

            # write into preallocated dz buffer
            dz[:H1]  = df * sigmoid_deriv(f)
            dz[H1:H2] = di * sigmoid_deriv(i)
            dz[H2:H3] = dg * tanh_deriv(g)
            dz[H3:]   = do * sigmoid_deriv(o)

            # nplace accumulation, no intermediate allocation
            dW  += np.outer(dz, xh)   # unavoidable alloc but outer is C-level
            db  += dz

            dxh        = self.W.T @ dz
            dx_seq[t]  = dxh[:self.input_size]
            dh         = dxh[self.input_size:]

        return {"dW": dW, "db": db}, dx_seq, dh, dc


class kNNAugmentedLSTM(LSTMCell):
    def __init__(self, *args, memory_capacity=4000, memory_k=8, memory_metric='euclidean', **kwargs):
        super().__init__(*args, **kwargs)
        dim_ratio = self.hidden_size // self.input_size
        self.memory = PerCellMemory(self.input_size, dim_ratio, capacity=memory_capacity, 
                       k=memory_k, metric=memory_metric)
        self.mem_gate = np.zeros(self.hidden_size)
        self.memory_k = memory_k
        self.memory_write_enabled = True
        self.cache = {}


    def memory_cell(self, f, c_regular, c_mem, i_mem, o_mem, g_mem, mem_mask):
        l_dim = f.shape[-1]
        scale = 1.0 / np.sqrt(l_dim)

        f_c = (f[:, np.newaxis] * c_mem + i_mem * g_mem)
        avg_m = (f_c * c_regular[:, np.newaxis])
        c_out = np.clip(avg_m, -10.0, 10.0)  # prevent overflow in tanh
        tanh_c = np.tanh(c_out)

        scores = np.where(mem_mask > 0, tanh_c, -1e9)
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores) * mem_mask
        weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

        out = (weights * o_mem).sum(axis=1)
        has_memory = (mem_mask.sum(axis=-1, keepdims=True) > 0).astype(np.float32)
        gate = sigmoid(self.mem_gate).reshape(self.hidden_size)
        gate_used = has_memory * gate + (1.0 - has_memory) * 1.0

        proper_out = np.dot(gate_used, c_regular[:, np.newaxis]).squeeze()
        dot_out = np.dot((1.0 - gate_used), out[:, np.newaxis]).sum(axis=1)
        cell_out = proper_out + dot_out
        return has_memory, gate, cell_out, tanh_c, out, weights

        
    def memory_cell_backward(self,
        dz, tanh_c_mem, dc, o_mem, mem_weights):
        dim_last = dz.shape[-1]
        scale = 1.0 / np.sqrt(dim_last)

        z_weights = (dz @ tanh_c_mem.T)
        dtanhz = (dz @ o_mem.T)

        multi = (dtanhz @ mem_weights)
        z_scores = (dtanhz - np.sum(multi, axis=-1, keepdims=True)) @ mem_weights
        z_scores = z_scores * scale

        dz_mem = (z_scores @ (tanh_deriv(tanh_c_mem) + dc).T).sum(axis=1)
        return dz_mem


    def _gate_and_mem_grads(self, dc_new):
        gate = self.cache['gate']
        has_memory = self.cache['has_memory']
        tanh_c = self.cache['tanh_c']
        h_out = self.cache['h_out']

        gate_a = (has_memory * gate + (1.0 - has_memory))
        gate_b = ((has_memory * (1.0 - gate))).sum(axis=1)

        c_local_out = np.dot(gate_a, dc_new)
        c_mem_out = c_local_out * gate_b[:, np.newaxis]

        return c_local_out, c_mem_out



    def forward(self, x_seq, h0=None, c0=None):
        T              = x_seq.shape[0]
        H              = self.hidden_size
        expected_input = self.input_size

        h = np.zeros(H) if h0 is None else h0.copy()
        c = np.zeros(H) if c0 is None else c0.copy()

        hs    = np.zeros((T, H))
        cs    = np.zeros((T, H))
        cache = []

        # preallocate xh buffer once only here.
        xh = np.empty(expected_input + H)
        for t in range(T):
            x = x_seq[t]
            if x.ndim == 0:
                x = x.reshape(1)
            if x.shape[0] < expected_input:
                x = np.pad(x, (0, expected_input - x.shape[0]))
            elif x.shape[0] > expected_input:
                x = x[:expected_input]
            
            xh[:expected_input] = x
            xh[expected_input:] = h

            z  = self.W @ xh + self.b
            H1, H2, H3 = H, H * 2, H * 3 

            f      = sigmoid(z[:H1])
            i      = sigmoid(z[H1:H2])
            g      = np.tanh(z[H2:H3])
            o      = sigmoid(z[H3:])
            
            c_new  = f * c + i * g
            c_regular = np.clip(c_new, -10.0, 10.0)  # prevent overflow in tanh

            c_mem, i_mem, o_mem, g_mem, mem_mask = self.memory.retrieve(f)
            has_memory, gate, cell_out, tanh_c, h_out, mem_weights = self.memory_cell(f, c_regular, c_mem, i_mem, o_mem, g_mem, mem_mask)
            
            cache.append((x.copy(), h_out.copy(), c_mem.copy(),
                        f, i_mem, g_mem, o_mem, cell_out, tanh_c, xh.copy()))

            self.cache['gate'] = gate
            self.cache['has_memory'] = has_memory
            self.cache['tanh_c'] = tanh_c
            self.cache['h_out'] = h_out
            self.cache['mem_weights'] = mem_weights

            h, c = h_out, cell_out
            hs[t] = h
            cs[t] = c

        if self.memory_write_enabled:
            self.memory.write(c_new, i, o, g)

        return hs, cs, cache
        

    def backward(self, dhs: np.ndarray, cache,
                dh_next=None, dc_next=None, T_limit=None):
        T = T_limit if T_limit is not None else len(cache)
        H = self.hidden_size

        dW     = np.zeros_like(self.W)
        db     = np.zeros_like(self.b)
        dh     = np.zeros(H) if dh_next is None else np.ascontiguousarray(dh_next.copy())
        dc     = np.zeros(H) if dc_next is None else np.ascontiguousarray(dc_next.copy())
        dx_seq = np.zeros((T, self.input_size))

        # preallocate dz buffer once
        dz     = np.empty((4 * H, self.memory_k))
        H1, H2, H3 = H, H * 2, H * 3

        for t in reversed(range(T)):
            x, h_prev, c_prev, f, i, g, o, c_new, tanh_c, xh = cache[t]

            dh_total = dhs[t] + dh

            do     = dh_total[:, np.newaxis] * tanh_c
            dtanhc = dh_total[:, np.newaxis] * o
            dc_deriv = dtanhc * tanh_deriv(tanh_c)
            dc_new = dc_deriv + dc[:, np.newaxis] if len(dc.shape) < 2 else dc

            c_local_out, dc_mem = self._gate_and_mem_grads(dc_new)

            df = dc_mem * c_prev
            di = dc_mem * g
            dg = dc_mem * i
            dc = dc_mem * f[:, np.newaxis]

            # write into preallocated dz buffer
            dz[:H1, :]  = (df * sigmoid_deriv(f)[:, np.newaxis])
            dz[H1:H2, :] = di * sigmoid_deriv(i)
            dz[H2:H3, :] = dg * tanh_deriv(g)
            dz[H3:, :]   = do * sigmoid_deriv(o)

            dz_mem = self.memory_cell_backward(
                dz, tanh_c, dc_mem, o, self.cache['mem_weights']
            )

            # nplace accumulation, no intermediate allocation
            dW  += np.outer(dz_mem, xh)   # unavoidable alloc but outer is C-level
            db  += dz_mem

            dxh        = self.W.T @ dz_mem
            dx_seq[t]  = dxh[:self.input_size]
            dh         = dxh[self.input_size:]

        return {"dW": dW, "db": db}, dx_seq, dh, dc        




# ─────────────────────────────────────────────
#  LSTM Network (cell + linear output head)
# ─────────────────────────────────────────────
class LSTMNetwork:
    def __init__(self, pipeline, input_size, hidden_size, output_size, seed=0):
        self.cell         = LSTMCell(input_size, hidden_size, seed)
        self.knn_cell     = kNNAugmentedLSTM(input_size=input_size, hidden_size=hidden_size, seed=seed)
        self.weight_shaper = GeometricWeightShaping(output_size, hidden_size)
        self.Wy           = None
        self.by           = np.zeros(output_size)
        self.pipeline     = pipeline
        self._trained     = False

        self._grad_norm_history = []  # for logging gradient health

    # forward method to calculate proper weight for prediction and training.
    def forward(self, x_seq):
        # also Wy init only here, removed from train_step
        if self.Wy is None:
            self.Wy = self.weight_shaper.weight_shaping(x_seq)
            expected_shape = (self.by.shape[0], self.cell.hidden_size)
            if self.Wy.shape != expected_shape:
                print(f'[⚠️] Wy shape {self.Wy.shape} != expected {expected_shape} '
                    f'— reshaping/reinitializing to avoid silent corruption')
                if self.Wy.size == np.prod(expected_shape):
                    self.Wy = self.Wy.reshape(expected_shape)
                else:
                    self.Wy = np.random.randn(*expected_shape) * 0.01

        hs, cs, cache = self.cell.forward(x_seq)
        preds = hs @ self.Wy.T + self.by   # (T, output_size)
        return preds, hs, cs, cache

    def forward_memory(self, x_seq):
        if self.Wy is None:
            self.Wy = self.weight_shaper.weight_shaping(x_seq)
            expected_shape = (self.by.shape[0], self.cell.hidden_size)
            if self.Wy.shape != expected_shape:
                print(f'[⚠️] Wy shape {self.Wy.shape} != expected {expected_shape} '
                    f'— reshaping/reinitializing to avoid silent corruption')
                if self.Wy.size == np.prod(expected_shape):
                    self.Wy = self.Wy.reshape(expected_shape)
                else:
                    self.Wy = np.random.randn(*expected_shape) * 0.01

        hs, cs, cache = self.knn_cell.forward(x_seq)
        preds = hs @ self.Wy.T + self.by   # (T, output_size)
        return preds, hs, cs, cache


    # calculate loss of MSE (Mean squared error.)
    def loss_mse(self, preds, targets, AMR):
        """
        Many-to-one loss: only the final timestep's prediction is
        compared against the single sequence-level target.
        """
        preds   = np.asarray(preds)
        targets = np.asarray(targets).reshape(-1)   # flatten to 1D regardless of input shape

        if preds.ndim == 1:
            preds = preds.reshape(1, -1)

        last_pred = preds[-1]   

        min_F = min(last_pred.shape[0], targets.shape[0])
        last_pred_aligned = last_pred[:min_F]
        targets_aligned   = targets[:min_F]

        diff = last_pred_aligned - targets_aligned
        loss = (1.0 - AMR) * np.mean(diff ** 2)

        dpreds = np.zeros_like(preds)
        dpreds[-1, :min_F] = diff / (min_F + 1e-8)

        return loss, dpreds

    # backward method for the network to calculate proper weights with cell backward
    def backward(self, dpreds, hs, cache):
        min_T  = min(dpreds.shape[0], hs.shape[0])
        dpreds = dpreds[:min_T]
        hs     = hs[:min_T]

        dWy = dpreds.T @ hs
        dby = dpreds.sum(axis=0)
        dhs = dpreds @ self.Wy

        # pass min_T directly, need to avoid creating a sliced list
        cell_grads, dx, _, _ = self.cell.backward(dhs, cache, T_limit=min_T)
        return cell_grads, {"dWy": dWy, "dby": dby}, dx

    def memory_backward(self, dpreds, hs, cache):
        min_T  = min(dpreds.shape[0], hs.shape[0])
        dpreds = dpreds[:min_T]
        hs     = hs[:min_T]

        dWy = dpreds.T @ hs
        dby = dpreds.sum(axis=0)
        dhs = dpreds @ self.Wy

        # pass min_T directly, need to avoid creating a sliced list
        cell_grads, dx, _, _ = self.knn_cell.backward(dhs, cache, T_limit=min_T)
        return cell_grads, {"dWy": dWy, "dby": dby}, dx

    # update ensured proper gradient clipping
    def update(self, cell_grads, out_grads, lr=1e-3, max_norm=5.0):
        all_grads = {**cell_grads, **out_grads}
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_grads.values()))
        clip_coef  = min(1.0, max_norm / (total_norm + 1e-6))

        if clip_coef < 1.0:
            for g in all_grads.values():
                g *= clip_coef   # single scale factor, preserves gradient direction here.

        self.cell.W -= lr * cell_grads["dW"]
        self.cell.b -= lr * cell_grads["db"]
        self.Wy     -= lr * out_grads["dWy"]
        self.by     -= lr * out_grads["dby"]

    # train step for each LSTM fitting method 
    def train_step(self, x_seq, targets, lr=1e-3, mode='knn', AMR=None, log_grad_health=True):
        # accept precomputed AMR 
        if AMR is None:
            AME = self.pipeline.AME_Encoder(x_seq)
            AMR = 1.0 / (1.0 + np.exp(-AME))

        if not mode == 'knn':
            preds, hs, cs, cache = self.forward(x_seq)
            loss, dloss          = self.loss_mse(preds, targets, AMR)
            cell_grads, out_grads, _ = self.backward(dloss, hs, cache)
        else:
            preds, hs, cs, cache = self.forward_memory(x_seq)
            loss, dloss          = self.loss_mse(preds, targets, AMR)
            cell_grads, out_grads, _ = self.memory_backward(dloss, hs, cache)

        if log_grad_health:
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in {**cell_grads, **out_grads}.values()))
            if not hasattr(self, '_grad_norm_history'):
                self._grad_norm_history = []
            self._grad_norm_history.append(float(grad_norm))

            # flag genuinely pathological training, not just noise
            if len(self._grad_norm_history) >= 10:
                recent = self._grad_norm_history[-10:]
                if np.mean(recent) < 1e-6:
                    print('[⚠️] LSTM gradient norm near-zero for 10 steps — '
                        'possible vanishing gradient, training may have stalled!')
                elif np.mean(recent) > 100.0:
                    print('[⚠️] LSTM gradient norm consistently large — '
                        'clipping is doing heavy lifting, consider lowering learning rate!')

        self.update(cell_grads, out_grads, lr)
        return loss, preds



# ─────────────────────────────────────────────
#  LSTM Engine
# ─────────────────────────────────────────────

class LSTMCustomizedEngine:
    """
    Wraps a trained LSTMNetwork and adds three confidence layers:

      Layer 1 — MC Dropout on hidden state (h)
                Perturbs h between timesteps — respects the cell's
                internal recurrence without touching W or b.
                Most faithful to this architecture.

      Layer 2 — Gate uncertainty
                Reads forget/input gate activations directly from cache.
                Low forget + high input = model is overwriting memory
                = structurally uncertain moment.

      Layer 3 — Prediction interval
                Built from validation residuals. Distribution-free,
                zero extra parameters, works on edge hardware.

    Usage:
        engine = LSTMEngine(model, dropout=0.1, n_samples=50)
        engine.calibrate(X_val, Y_val)
        result = engine.predict(x_seq, label_bins=None)
    """

    def __init__(self, pipeline: Any, dropout: float = 0.1,
                 n_samples: int = 50):
        self.pipeline = pipeline
        self.model     = LSTMNetwork(self, input_size=1, hidden_size=32, output_size=1)
        self.dropout   = dropout
        self.n_samples = n_samples
        self.residual_std  = None   # set by calibrate()
        self.residual_mean = None

    # ── calibrate on validation set ──────────
    def calibrate_residual(self, X_val, Y_val):
        if len(X_val) == 0:
            self.residual_mean = 0.0
            self.residual_std  = 1.0
            return

        confidence_errors = []

        for j in range(len(X_val)):
            preds, _, _, _ = self.model.forward(X_val[j])

            # preds is (T, output_size); only the LAST timestep is
            # the actual prediction, matching fit_stm/loss_mse's many-to-one
            # convention
            last_pred = preds[-1] if preds.ndim > 1 else preds
            pred_val  = float(last_pred[0]) if last_pred.ndim > 0 else float(last_pred)

            # target_j is a bare scalar confidence value,
            # — np.atleast_1d then take first
            # element, no len()/ndim branching needed
            target_j  = np.atleast_1d(np.asarray(Y_val[j], dtype=np.float64))
            true_val  = float(target_j[0])

            # irect residual, pred_val is
            # already a continuous regression output being compared
            # against a continuous confidence target.
            err = abs(pred_val - true_val)
            confidence_errors.append(err)

        errors = np.array(confidence_errors)

        # IQR outlier removal
        # 1D error array regardless of what generated it
        q25, q75 = np.percentile(errors, [25, 75])
        iqr      = q75 - q25
        mask     = (errors >= q25 - 1.5 * iqr) & (errors <= q75 + 1.5 * iqr)
        clean    = errors[mask] if mask.sum() > 0 else errors

        self.residual_mean = float(clean.mean())
        self.residual_std  = float(max(clean.std(), 1e-6))

        if len(X_val) < 20:
            self.residual_std = max(self.residual_std, 0.1)
            print(f'[!] Small val set ({len(X_val)} samples) — flooring σ to 0.1')

        self.calibration_coverage  = float(mask.mean())
        self.n_calibration_samples = len(X_val)

        print(f"[=] Calibrated: residual μ={self.residual_mean:.4f} "
            f"σ={self.residual_std:.4f} "
            f"coverage={self.calibration_coverage:.1%} "
            f"n={self.n_calibration_samples}")

    # ── MC dropout forward ────────────────────
    def _mc_forward(self, x_seq: np.ndarray) -> np.ndarray:

        eps = 1e-5
        T            = x_seq.shape[0]
        H            = self.model.cell.hidden_size
        expected_input = self.model.cell.input_size
        p            = self.dropout
        cell         = self.model.cell
        W            = cell.W
        b            = cell.b
        H1, H2, H3  = H, H * 2, H * 3   # slice boundaries precomputed

        # Wy check once before loop
        if self.model.Wy is None:
            self.model.Wy = self.model.weight_shaper.weight_shaping(x_seq)
        Wy = self.model.Wy
        by = self.model.by

        h     = np.zeros(H)
        c     = np.zeros(H)
        xh    = np.empty(expected_input + H)  # preallocate concat buffer
        preds = np.empty(T)                   # preallocate output

        # precompute dropout scale factor
        inv_keep = 1.0 / (1.0 - p) + eps

        for t in range(T):
            x = x_seq[t]

            # shape alignment
            if x.ndim == 0:
                x = x.reshape(1)
            if x.shape[0] < expected_input:
                x = np.pad(x, (0, expected_input - x.shape[0]))
            elif x.shape[0] > expected_input:
                x = x[:expected_input]

            # need to write into preallocated buffer instead of np.concatenate
            xh[:expected_input] = x
            xh[expected_input:] = h

            z = W @ xh + b                    # (4H,)

            # direct slices instead of method calls
            f      = sigmoid(z[:H1])
            i      = sigmoid(z[H1:H2])
            g      = np.tanh(z[H2:H3])
            o      = sigmoid(z[H3:])

            c      = f * c + i * g
            tanh_c = np.tanh(c)
            h      = o * tanh_c

            # precomputed inv_keep, inplace mask application
            mask = (np.random.rand(H) > p) * inv_keep
            h   *= mask

            preds[t] = (h @ Wy.T + by)[0]

        return preds   # (T,)

    # ── gate uncertainty for LSTM prediction──────────────────────
    def _gate_uncertainty(self, x_seq: np.ndarray, AMR: float) -> np.ndarray:
        """
        Structural uncertainty from gate activations.
        Vectorized — no Python loop over timesteps.
        """
        _, _, cache = self.model.cell.forward(x_seq)

        # cache[t] = (x, h_prev, c_prev, f, i, g, o, c_new, tanh_c, xh)
        # need to extract f and i directly as stacked arrays — shape (T, H)
        T = len(cache)
        
        if T == 0:
            return np.array([0.0])

        # vectorized extraction — one pass instead of T unpacks
        f_all = np.empty((T, cache[0][3].shape[0]))  # (T, H)
        i_all = np.empty((T, cache[0][4].shape[0]))  # (T, H)

        for t, entry in enumerate(cache):
            f_all[t] = entry[3]   # forget gate
            i_all[t] = entry[4]   # input gate

        # vectorized computation — no per-timestep Python arithmetic
        forget_instability = 1.0 - f_all.mean(axis=1)   # (T,)
        input_activity     = i_all.mean(axis=1)           # (T,)

        # precompute scalar factor once
        scale = 1.0 - AMR
        gate_uncertainty = scale * (forget_instability + input_activity)

        return np.clip(gate_uncertainty, 0.0, 1.0)      

    # empirical quantiles from actual residuals
    def calibrate(self, X_val, Y_val):
        if len(X_val) == 0:
            self.residual_mean = 0.0
            self.residual_std  = 1.0
            self.quantiles     = {}
            return

        all_errors = []
        for j in range(len(X_val)):
            preds, _, _, _ = self.model.forward(X_val[j])
            y         = Y_val[j]
            last_pred = preds[-1] if preds.ndim > 1 else preds
            pred_vals = float(last_pred[0]) if last_pred.ndim > 0 else float(last_pred)

            true_vals = float(np.atleast_1d(np.asarray(Y_val[j], dtype=np.float64))[0])
        
            all_errors.append(pred_vals - true_vals)

        residuals = np.array(all_errors)
        n         = len(residuals)

        self.residual_mean = float(residuals.mean())
        self.residual_std  = float(max(residuals.std(), 1e-6))

        # adapt confidence levels to sample size
        # small n → only compute what's statistically supportable
        if n < 20:
            # only 90% interval is reliable — use 10th/90th percentile
            # wide enough to be honest about uncertainty
            levels = [10.0, 90.0]
            p = np.percentile(residuals, levels)
            self.quantiles = {
                0.90: (float(p[0]), float(p[1])),
                0.95: (float(p[0]), float(p[1])),  # same as 90% — honest, not fake precision
                0.99: (float(p[0]), float(p[1])),
            }
            print(f'[!] n={n} too small for tail quantiles — '
                f'using 10th/90th for all intervals')

        elif n < 50:
            # 90% and 95% supportable, 99% not reliable
            levels = [2.5, 5.0, 95.0, 97.5]
            p = np.percentile(residuals, levels)
            self.quantiles = {
                0.90: (float(p[1]), float(p[2])),
                0.95: (float(p[0]), float(p[3])),
                0.99: (float(p[0]), float(p[3])),  # same as 95% — honest
            }
            print(f'[!] n={n} insufficient for 99% interval — '
                f'using 95% as proxy')

        else:
            # full precision justified
            levels = [0.5, 2.5, 5.0, 95.0, 97.5, 99.5]
            p = np.percentile(residuals, levels)
            self.quantiles = {
                0.90: (float(p[2]), float(p[3])),
                0.95: (float(p[1]), float(p[4])),
                0.99: (float(p[0]), float(p[5])),
            }

        # floor std on small n
        if n < 20:
            self.residual_std = max(self.residual_std, 0.1)

        print(f"[=] Calibrated: μ={self.residual_mean:.4f} "
            f"σ={self.residual_std:.4f} "
            f"n={n} "
            f"90%=[{self.quantiles[0.90][0]:.4f}, {self.quantiles[0.90][1]:.4f}]")

    # interval to calculate prediction interval from MC mean + empirical quantiles
    def _interval(self, mc_mean, confidence_level):
        # flatten mc_mean safely — handles scalar, 0-d array, or 1-d array
        mc_scalar = float(np.asarray(mc_mean).flat[0])

        if confidence_level not in self.quantiles:
            available = sorted(self.quantiles.keys())
            if not available:
                return mc_scalar - self.residual_std, mc_scalar + self.residual_std
            confidence_level = min(available, key=lambda k: abs(k - confidence_level))
            print(f'[!] Confidence level not found, using closest: {confidence_level}')

        lo_bias, hi_bias = self.quantiles[confidence_level]

        lo = mc_scalar + float(np.asarray(lo_bias).flat[0])
        hi = mc_scalar + float(np.asarray(hi_bias).flat[0])

        if lo > hi:
            lo, hi = hi, lo

        return lo, hi

    # MC sample counting for label confidence (last timestep)
    def _label_confidence_empirical(self, mc_samples_last, label_bins):
        """
        mc_samples_last : (n_samples,) — raw MC draws at last timestep
        label_bins      : {"Good": (0, 35), "Moderate": (35, 75), ...}
        """
        n = len(mc_samples_last)
        if n == 0:
            return {name: 0.0 for name in label_bins}

        names  = list(label_bins.keys())
        bounds = np.array(list(label_bins.values()))  # (n_bins, 2)

        # vectorized — broadcast (n_samples,) against (n_bins, 2)
        # samples shape: (1, n_samples), bounds shape: (n_bins, 1)
        samples = mc_samples_last[np.newaxis, :]          # (1, n_samples)
        lo      = bounds[:, 0, np.newaxis]                # (n_bins, 1)
        hi      = bounds[:, 1, np.newaxis]                # (n_bins, 1)

        hits = ((samples >= lo) & (samples < hi)).sum(axis=1)  # (n_bins,)
        probs = hits / n                                        # (n_bins,)

        return dict(zip(names, probs.tolist()))

    # LSTM training loop with confidence layers integrated into the loss and validation monitoring.
    def fit_stm(self, X, Y, epochs=50, hidden=32, lr=5e-3, seq_len=20, print_every=5, mode='knn'):
        print("[= =] Training LSTM with confidence layers "
            "(MC dropout + gate uncertainty + prediction intervals)")

        eps = 1e-3
        model = self.model

        X = np.asarray(X)
        Y = np.asarray(Y)

        if Y.ndim == 1 and np.issubdtype(Y.dtype, np.integer):
            unique_vals = np.unique(Y)
            if len(unique_vals) <= 20 and np.array_equal(unique_vals, unique_vals.astype(int)):
                print(f'[⚠️] fit_stm: Y looks like raw class indices '
                    f'(unique values: {unique_vals}) — LSTM regression on '
                    f'unbounded class integers is semantically wrong. '
                    f'Pass a genuine continuous target (confidence score or '
                    f'normalized value) instead.')

        AME = self.pipeline.AME_Encoder(X)
        AMR = 1.0 / (1.0 + np.exp(-AME))

    
        split_ratio = self.pipeline.confidence_threshold
        if not (0.0 < split_ratio < 1.0):
            split_ratio = 0.8   # sane default if confidence_threshold is out of range

        n_samples = len(X)
        if n_samples < 2:
            # too little data to split at all — train on everything,
            # explicitly skip validation rather than pretending to split
            n_train = n_samples
            print(f'[⚠️] fit_stm: only {n_samples} sample(s) — cannot create a '
                f'validation split, training on all data, validation will be skipped')
        else:
            # proper fraction-based split, clamped so BOTH sides
            # always have at least 1 sample when n_samples >= 2
            n_train = int(round(split_ratio * n_samples))
            n_train = max(1, min(n_train, n_samples - 1))

        X_tr, Y_tr = X[:n_train], Y[:n_train]
        X_te, Y_te = X[n_train:], Y[n_train:]
        if len(X_te) == 0:
            print('[⚠️] fit_stm: no validation samples remain after split — skipping validation loss reporting this run')

        idx = np.arange(n_train)
        val_loss = 0.0   

        training_not_allowed = AMR < 0.0 or AMR > 1.0 or np.isnan(AMR) or np.isinf(AMR) or n_samples < 2
        if training_not_allowed:
            print(f'[⚠️] fit_stm: training not allowed — AMR={AMR:.4f}, n_samples={n_samples} — skipping training loop, proceeding to calibration if possible')
        else:
            for epoch in range(1, epochs + 1):
                np.random.shuffle(idx)
                epoch_loss = 0.0
                for j in idx:
                    loss, _ = model.train_step(X_tr[j], Y_tr[j], lr=lr, AMR=AMR)
                    epoch_loss += loss
                epoch_loss /= (n_train + eps)

                is_last_epoch = (epoch == epochs)
                if epoch % print_every == 0 or epoch == 1 or is_last_epoch:
                    if len(X_te) > 0:
                        val_loss = 0.0
                        for j in range(len(X_te)):
                            preds, _, _, _ = model.forward(X_te[j])
                            target_j = np.atleast_1d(np.asarray(Y_te[j], dtype=np.float64))

                            if preds.ndim == 1:
                                preds = preds.reshape(-1, 1)

                            last_pred = preds[-1] if preds.ndim > 1 else preds
                            last_target = target_j

                            min_F = min(preds.shape[0], last_target.shape[0])

                            val_loss += AMR * np.mean((last_pred[:min_F] - last_target[:min_F]) ** 2)
                        val_loss /= len(X_te)
                    else:
                        val_loss = float('nan')

                    print(f"[=] Epoch {epoch:>4}/{epochs}  "
                        f"[=] train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

            print("[=] Training complete!")
            print(f"[=] Final val loss: {val_loss:.6f}")   # always reflects
                                                            # the most recent real
                                                            # computation, never stale

        print('===== CALIBRATION METHOD =====')
        # calibration runs on the
        # actual intended validation set
        if len(X_te) > 0:
            self.calibrate_residual(X_te, Y_te)
        else:
            print('[⚠️] Skipping calibration — no validation samples available')

    # get optimal lstm samples amount for the model to process
    def lstm_optimal_samples(self, engine, x_seq, tolerance=0.005, max_n=500):
        """
        Run increasing n_samples until std estimate stabilizes.
        Stable = change in std < tolerance between consecutive checks.
        """
        prev_std = None
        for n in range(10, max_n, 10):
            samples = np.stack([
                engine._mc_forward(x_seq) for _ in range(n)
            ])
            current_std = samples.std(axis=0).mean()
            if prev_std is not None:
                delta = abs(current_std - prev_std)
                print(f"  n={n:>4}  std={current_std:.5f}  delta={delta:.5f}")
                if delta < tolerance:
                    print(f"  → Converged at n={n}")
                    return n
            prev_std = current_std
        return max_n

    # derive local bins for flexibility in scarce dataset
    def derive_bins_from_data(self, y_values, n_bins=4, labels=None,
                            value_range=None):
        """
        value_range: optional (min, max) tuple describing the theoretical
        bounds of y_values (e.g. (0.0, 1.0) for confidence scores). When
        given, the top bin's upper edge is clamped to this range instead
        of using a multiplicative headroom, which badly distorts bounded
        data like probabilities/confidences.
        """
        unique_vals = np.unique(y_values)

        if len(unique_vals) <= 2:
            if labels is None:
                labels = ["Negative", "Positive"]
            return {
                labels[0]: (float(unique_vals[0]) - 1e-6, float(unique_vals[0]) + 1e-6),
                labels[1]: (float(unique_vals[-1]) - 1e-6, float(unique_vals[-1]) + 1e-6),
            }

        dominant_val  = float(np.percentile(y_values, 50))
        dominant_frac = (y_values == dominant_val).mean()

        # computed headroom once, consistently, respecting value_range
        def _top_edge(boundary_max):
            if value_range is not None:
                return float(value_range[1])   # clamp to known theoretical max
            # small ADDITIVE headroom, — safe for any scale
            span = boundary_max - float(np.percentile(y_values, 0))
            return float(boundary_max + max(span * 0.1, 1e-3))

        if dominant_frac > 0.5:
            non_dominant = y_values[y_values != dominant_val]
            percentiles  = np.linspace(0, 100, n_bins)
            boundaries   = np.percentile(non_dominant, percentiles)

            if labels is None:
                labels = ["Base"] + [f"Level_{i+1}" for i in range(n_bins-1)]    

            base_lo = float(value_range[0]) if value_range else -1e-6
            bins = {"Base": (base_lo, float(boundaries[0]))}
            for i in range(n_bins - 1):
                lo = boundaries[i]
                hi = boundaries[i+1] if i < n_bins-2 else _top_edge(boundaries[i+1])
                bins[labels[i+1]] = (float(lo), float(hi))
            return bins

        percentiles = np.linspace(0, 100, n_bins + 1)
        boundaries  = np.percentile(y_values, percentiles)
        if labels is None:
            labels = [f"Level_{i+1}" for i in range(n_bins)]

        bins = {}
        for i in range(n_bins):
            lo = boundaries[i]
            hi = boundaries[i+1] if i < n_bins-1 else _top_edge(boundaries[i+1])
            bins[labels[i]] = (float(lo), float(hi))
        return bins

    # ── main predict function for the whole network ─────────────────────────
    def predict(self, x_seq: np.ndarray,
                label_bins: dict = None,
                confidence_level: float = 0.90) -> Any:
        """
        Full confidence-aware prediction.

        Args:
            x_seq          : (T, input_size)
            label_bins     : optional dict defining label thresholds
                             e.g. {"Low": (-1, -0.33),
                                   "Mid": (-0.33, 0.33),
                                   "High": (0.33, 1.0)}
            confidence_level: for prediction interval (default 90%)

        Returns dict with:
            prediction     : point estimate (T,)
            mc_mean        : MC dropout mean (T,)
            mc_std         : MC dropout std  (T,)
            mc_confidence  : per-timestep confidence in [0,1]
            gate_uncertainty: structural uncertainty (T,)
            interval_low   : lower bound of prediction interval (T,)
            interval_high  : upper bound of prediction interval (T,)
            label_confidence: {label: probability} if label_bins given
            overall        : single scalar confidence for last timestep
        """
        # ── point prediction ──────────────────
        preds_clean, _, _, _ = self.model.forward(x_seq)
        AME = self.pipeline.AME_Encoder(x_seq)  # geometric complexity scalar
        AMR = 1.0 / (1.0 + np.exp(-AME))  # abstract modelling rate
        point = preds_clean[:, 0]   # (T,)

        if np.isnan(AMR) or np.isinf(AMR) or AMR <= 1e-10:
            AMR = self.pipeline.confidence_threshold + 1e-5

        # ── MC dropout sampling ───────────────
        samples = np.stack([
            self._mc_forward(x_seq) for _ in range(self.n_samples)
        ])  # (n_samples, T)


        mc_mean = samples.mean(axis=0)   # (T,)
        mc_std  = samples.std(axis=0)    # (T,)

        # confidence: tight distribution = high confidence
        # normalize std by typical residual std so scale is meaningful
        normalized_std = mc_std / (self.residual_std + 1e-8)
        mc_confidence  = np.exp(-normalized_std)   # (T,) in (0,1]

        # ── gate uncertainty ──────────────────
        gate_unc = self._gate_uncertainty(x_seq, AMR)   # (T,)

        # ── prediction interval ───────────────
        total_std = np.sqrt(mc_std**2 + self.residual_std**2)        
        # compute interval for every timestep — always returns (T,) arrays
        interval_low  = np.empty(len(mc_mean))
        interval_high = np.empty(len(mc_mean))
        for t in range(len(mc_mean)):
            interval_low[t], interval_high[t] = self._interval(mc_mean[t], confidence_level)

        # ── label confidence (last timestep) ──
        label_conf = None
        if label_bins is not None:
            label_conf = self._label_confidence_empirical(samples[:, -1], 
                          label_bins)

            # renormalize
            total_p = sum(label_conf.values()) + 1e-8
            label_conf = {k: v/total_p for k, v in label_conf.items()}

        # ── overall scalar confidence ─────────
        # weighted combination of MC confidence and gate stability
        gate_stability = 1.0 - gate_unc[-1]           # high = stable
        overall = (1.0 - AMR) * mc_confidence[-1] + \
                  self.pipeline.confidence_threshold * gate_stability     

        return {
            "prediction"      : point,
            "mc_mean"         : mc_mean,
            "mc_std"          : mc_std,
            "mc_confidence"   : mc_confidence,
            "gate_uncertainty": gate_unc,
            "interval_low"    : interval_low,
            "interval_high"   : interval_high,
            "label_confidence": label_conf,
            "overall"         : overall,
        }
        
    # ─────────────────────────────────────────────
    #  Architecture summary helper to visualize results
    # ─────────────────────────────────────────────
    def architectural_summary(self, model: LSTMNetwork):
        H = model.cell.hidden_size
        I = model.cell.input_size
        O = model.Wy.shape[0]
        W_params  = model.cell.W.size + model.cell.b.size
        Wy_params = model.Wy.size + model.by.size
        total     = W_params + Wy_params

        print("\n┌─────────────────────────────────────────┐")
        print("  │          LSTM Architecture Summary      │")
        print("  ├─────────────────────────────────────────┤")
        print(f" │  Input  size   : {I:<24}                │")             
        print(f" │  Hidden size   : {H:<24}                │")
        print(f" │  Output size   : {O:<24}                │")
        print(f" │  LSTM   params : {W_params:<24,}        │")
        print(f" │  Linear params : {Wy_params:<24,}       │")
        print(f" │  Total  params : {total:<24,}           │")
        print("  └─────────────────────────────────────────┘")



if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim = 32
    n_points = 3000
    data = rng.normal(size=(n_points, dim)).astype(np.float32)

    index = HNSW(dim=dim, M=16, ef_construction=200, metric="euclidean", seed=0)
    for v in data:
        index.insert(v)

    def brute_force(query, k):
        dists = np.linalg.norm(data - query, axis=1)
        idx = np.argsort(dists)[:k]
        return set(idx.tolist())

    k = 10
    n_queries = 20
    recalls = []
    for _ in range(n_queries):
        q = rng.normal(size=dim).astype(np.float32)
        approx = {n for n, _ in index.search(q, k=k, ef=100)}
        exact = brute_force(q, k)
        recalls.append(len(approx & exact) / k)

    print(f"Indexed {n_points} points, dim={dim}")
    print(f"Average recall@{k} over {n_queries} queries: {np.mean(recalls):.3f}")
    

        



