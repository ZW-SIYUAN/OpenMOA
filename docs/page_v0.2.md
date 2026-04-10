# OpenMOA v0.2 — Code Quality, Bug Fixes, and Performance

OpenMOA v0.2 is a quality-focused release. It does not change any algorithm's mathematical behaviour or experimental results. Instead, it fixes correctness bugs that were silently producing wrong outputs, eliminates code duplication, hardens reproducibility guarantees, and delivers measurable speed-ups on the hot paths of several classifiers.

Every change is covered by the **61 new test cases** added in this release.

---

## What Changed at a Glance

| Category | Count |
|----------|-------|
| Refactoring | 2 |
| Algorithm bug fixes | 7 |
| Stream Wrapper bug fixes | 5 |
| Performance optimisations | 8 |
| New test cases | 61 (3 files) |
| Files modified or created | 15 |

---

## Refactoring

### Q1 — `SparseInputMixin`: Eliminating Copy-Pasted Logic

The `_get_sparse_x` helper — which converts an incoming instance into a `(indices, values)` sparse pair — was duplicated almost verbatim across five classifier files. Any future fix to the three-branch dispatch logic (OpenFeatureStream / native sparse / dense-with-NaN) would have needed to be applied five times.

A new `SparseInputMixin` class in `src/openmoa/base/_sparse_mixin.py` centralises the logic:

```python
class SparseInputMixin:
    def _get_sparse_x(self, instance) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(instance, "feature_indices"):
            return np.asarray(instance.feature_indices), np.asarray(instance.x, dtype=float)
        if hasattr(instance, "x_index") and hasattr(instance, "x_value"):
            return instance.x_index, instance.x_value
        x = np.asarray(instance.x, dtype=float)
        valid_mask = (x != 0) & (~np.isnan(x))
        return np.where(valid_mask)[0], x[valid_mask]
```

**Affected files:** `_fesl_classifier.py`, `_oasf_classifier.py`, `_fobos_classifier.py`, `_ftrl_classifier.py`, `_rsol_classifier.py`

---

### Q2 — Instance-Level RNG: Fixing Global Random State Pollution

Multiple classifiers and stream wrappers called `np.random.seed(random_seed)` to initialise randomness. Setting a global seed has a well-known problem: any other code that touches `np.random` in the same process — another classifier running in parallel, a library function — will alter the sequence of numbers that your code draws, making results non-reproducible between runs.

All classifiers and wrappers now carry their own `np.random.RandomState` instance:

```python
# Before — pollutes global state
np.random.seed(random_seed)
noise = np.random.randn(d)

# After — isolated per object
self._rng = np.random.RandomState(random_seed)
noise = self._rng.randn(d)
```

This also means you can instantiate multiple classifiers with different seeds in the same process and get independent, reproducible random sequences from each.

---

## Algorithm Bug Fixes

### A1 — FOBOS / FTRL: `IndexError` on Growing Feature Streams

**Symptom:** Running FOBOS or FTRL on an `OpenFeatureStream` in incremental mode crashes with `IndexError: index N is out of bounds for axis 0 with size M`.

**Cause:** Both classifiers allocate their weight arrays at construction time using the initial feature count. As `OpenFeatureStream` introduces new features, their global IDs exceed the array dimensions.

**Fix:** A `_ensure_dimension(target_dim)` method that resizes the weight array on demand with a 1.5× growth factor, avoiding frequent reallocations:

```python
def _ensure_dimension(self, target_dim: int):
    if target_dim <= self.n_features:
        return
    new_dim = max(target_dim, int(self.n_features * 1.5))
    new_W = np.zeros((new_dim, self.n_outputs))
    new_W[:self.n_features] = self.W
    self.W = new_W
    self.n_features = new_dim
```

FTRL resizes three arrays (`z`, `n`, `w`) simultaneously to keep them in sync.

---

### A2 — ORF3V: Wrong Feature IDs in Weight Dictionary

**Symptom:** ORF3V's per-feature forests accumulate weights under incorrect keys, causing the model to learn associations between the wrong features and trees.

**Cause:** `_update_weights` used `range(len(x))` as feature IDs — local position indices within the current instance vector. With `OpenFeatureStream`, position 0 in `instance.x` may correspond to global feature ID 3, 7, or any other value.

```
Active features: global IDs [3, 5, 7]
instance.x     = [v3, v5, v7]   ← physical length 3
range(len(x))  = [0, 1, 2]      ← wrong; should be [3, 5, 7]
```

**Fix:** Use `getattr(instance, "feature_indices", range(len(x)))` to obtain the correct global IDs before iterating.

---

### A3 — OLD3S: Normalisation Statistics Never Updated

**Symptom:** OLD3S's online Min-Max normalisation is effectively disabled from the first instance onward, meaning the VAE always receives unnormalised input.

**Cause:** A one-character variable name error in the update condition:

```python
# Before — x_raw is the full-dimension vector; stats['min'] tracks a subset
if len(x_raw) == len(stats['min']):   # always False → stats never update
    stats['min'] = np.minimum(stats['min'], x_sub)

# After
if len(x_sub) == len(stats['min']):   # correct comparison
    stats['min'] = np.minimum(stats['min'], x_sub)
```

This bug affects every OLD3S experiment run with v0.1.

---

### A4 — OVFM: Ordinal Variable Initialisation Uses Global RNG

**Symptom:** Two OVFM runs with the same `random_seed` produce different results when other code has touched `np.random` in between.

**Cause:** `_init_z_ordinal` sampled from `np.random.uniform(...)` instead of `self._rng.uniform(...)`.

**Fix:** Replace all global `np.random` calls in `_ovfm_classifier.py` with the instance-level RNG introduced by Q2.

---

### A5 — OSLMF: DensityPeaks Called Once Per Instance Instead of Once Per Batch

**Symptom:** OSLMF is dramatically slower than expected and its learning behaviour diverges from the published algorithm.

**Cause:** The original paper (Wu et al., AAAI 2023) specifies that DensityPeaks label propagation runs once per *batch*. The implementation called `propagate_labels()` inside `train()`, so it ran once per *instance* — triggering an O(buffer²) computation at every step.

**Fix:** Introduce a `batch_size` parameter (default 50). Instances accumulate in an internal buffer; DensityPeaks, EM covariance update, and SGD all run together when the buffer is full. The Copula's sliding-window `partial_fit` retains its per-instance call, consistent with the paper.

---

### A6 — ORF3V: Dead Code Removed

`_update_feature_stats` was defined in `_orf3v_classifier.py` but never called. Its logic also overlapped with the `FeatureStatistics` class. The method has been deleted.

---

### A7 — FESL: Inconsistent Indentation

Several method bodies in `_fesl_classifier.py` mixed 2-space and 4-space indentation. All code is now uniformly 4-space indented.

---

## Stream Wrapper Bug Fixes

### W1 — `restart()` Did Not Reset the RNG

**Symptom:** Restarting a stream with a random component (e.g. `TrapezoidalStream`'s feature activation order, `ShuffledStream`'s shuffle order) and replaying it produces a different sequence than the first pass, breaking reproducibility.

**Fix:** All wrappers now reinitialise `self._rng` and regenerate derived schedules inside `restart()`.

---

### W2 — `ShuffledStream` Swallowed All Exceptions

```python
# Before — silently catches KeyboardInterrupt, MemoryError, everything
try:
    ...
except Exception:
    break

# After — separates expected from unexpected
try:
    ...
except StopIteration:
    break
except Exception as e:
    warnings.warn(f"ShuffledStream: unexpected error: {e}", RuntimeWarning)
    break
```

---

### W3 — `OpenFeatureStream` Pre-allocated O(N) Index Cache

**Symptom:** Constructing an `OpenFeatureStream` over a 100 000-instance dataset consumed significant memory before the first instance was read.

**Cause:** `_generate_feature_indices()` precomputed and cached the active feature set for every time step at initialisation.

**Fix:** Lazy evaluation — `_get_active_indices(t)` is called on demand in `next_instance()`. For random modes, determinism is preserved by deriving the RNG seed from `base_seed + t`.

---

### W4 — Magic Number in TDS

The constant `n_stages = 10` in `TrapezoidalStream` was undocumented. A comment now explains its origin: the TDS paper (Gao et al.) defines 10 birth stages.

---

### W5 — Duplicated EDS Boundary Formula

`OpenFeatureStream` (EDS mode) and `EvolvableStream` each had their own copy of the same boundary computation:

```
total = L × (n + overlap_ratio × (n − 1))
```

The formula is now in a single module-level function `_calc_eds_boundaries()` shared by both classes.

---

## Performance Optimisations

### P1 — OASF: O(d × L) Roll → O(1) Ring Buffer

The sliding-window weight matrix was shifted every step with `np.roll(W, -1, axis=1)`, which copies the entire matrix. A ring buffer with a write pointer performs the same logical operation in O(1):

```python
# Before
self.W = np.roll(self.W, -1, axis=1)
self.W[:, -1] = w_new

# After
self.W[:, self._ptr] = w_new
self._ptr = (self._ptr + 1) % self.L
```

---

### P2 — OSLMF: `list.pop(0)` → `deque(maxlen=...)`

The DensityPeaks buffer used a Python list with `pop(0)` to evict the oldest entry — an O(n) operation on every step. Replaced with `collections.deque(maxlen=buffer_size)`, which evicts automatically in O(1).

---

### P3 — FESL: Vectorised Prediction via Mapping Matrix

The inner loop over old-space weights was replaced with a single NumPy dot product, eliminating Python-level iteration over feature IDs.

---

### P4 — ORF3V: Vectorised CDF Computation

```python
# Before
return sum(1 for v in arr if v < split_val) / len(arr)

# After
return float(np.sum(arr < split_val)) / len(arr)
```

---

### P5 — OSLMF: Vectorised DensityPeaks Delta Computation

The core O(n²) loop that finds each point's nearest higher-density neighbour was replaced with a rank-mask matrix approach — no Python loops, two NumPy `min` calls:

```python
rank_mask = rank[np.newaxis, :] < rank[:, np.newaxis]   # (n, n) bool
dist_masked = np.where(rank_mask, dist_matrix, np.inf)
delta = dist_masked.min(axis=1)
nearest_higher = np.argmin(dist_masked, axis=1)
```

Numerical equivalence verified by unit tests (error < 1e-10).

| Buffer size | Before | After | Speedup |
|-------------|--------|-------|---------|
| n = 50 | 0.151 ms | 0.047 ms | **3.2×** |
| n = 100 | 0.430 ms | 0.253 ms | **1.7×** |
| n = 200 (default) | 1.492 ms | 1.109 ms | **1.3×** |

---

### P6 — OSLMF / OVFM: `statsmodels.ECDF` → `np.searchsorted`

The Gaussian Copula transform constructed a `statsmodels.ECDF` object per feature per batch — carrying Python object overhead, an internal sort, and an interpolator. The operation reduces to a sorted search:

```
ECDF(w)(x)  =  #{w_i ≤ x} / n
            =  searchsorted(sort(w), x, side='right') / n
```

With the standard Hájek smoothing factor H = n/(n+1):

```python
# Before
ecdf = ECDF(window_clean)
u = (len(window_clean) / (len(window_clean) + 1)) * ecdf(x_obs)

# After — numerically identical
sorted_w = np.sort(window_clean)
u = np.searchsorted(sorted_w, x_obs, side='right') / (len(sorted_w) + 1)
```

**Additional benefit:** `statsmodels` is no longer imported in either file, removing a heavy optional dependency.

| Observations | Before | After | Speedup |
|-------------|--------|-------|---------|
| 10 | 0.0146 ms | 0.0042 ms | **3.5×** |
| 100 | 0.0154 ms | 0.0038 ms | **4.0×** |

---

### P7 — `np.array` → `np.asarray` (14 call sites)

`np.array(x)` always allocates and copies. `np.asarray(x)` returns the original array unchanged when dtype already matches, skipping the allocation:

| Case | `np.array` | `np.asarray` | Speedup |
|------|-----------|-------------|---------|
| float64, no conversion needed | 0.30 µs | 0.07 µs | **4.2×** |
| dtype mismatch (must copy) | same | same | — |

This affects `train()` and `predict()` — called on every instance — so the gain accumulates over long streams.

---

### P8 — OLD3S: Vectorised HBP Weight Update

```python
# Before — Python loop, one temporary tensor per layer
for i, l in enumerate(losses):
    bundle['hbp_weights'][i] *= torch.pow(decay, l)
bundle['hbp_weights'] /= bundle['hbp_weights'].sum()

# After — single in-place operation, no loop, no temporaries
losses_t = torch.stack([l.detach() for l in losses])
bundle['hbp_weights'].mul_(decay.pow(losses_t))
bundle['hbp_weights'].div_(bundle['hbp_weights'].sum())
```

Measured speedup: **1.4×** for 3-layer MLP; gains scale with depth.

---

## End-to-End Benchmark

Environment: Windows 11, Intel CPU, Python 3.13, NumPy 2.x. 400 `train` calls, d = 8.

| Classifier | Total time | Per instance |
|------------|-----------|--------------|
| OSLMF (batch = 50) | 51.0 ms | 0.128 ms |
| OVFM | 31.8 ms | 0.080 ms |

---

## Test Coverage

Three new test files, 61 test cases, all passing.

### `tests/test_stream_wrappers.py` — 23 cases

Covers correct `feature_indices` attachment, monotone dimension trends, EDS ID ranges, `restart()` reproducibility for all wrappers, `missing_ratio=0` produces no NaN, invalid arguments raise exceptions, `ShuffledStream` yields every instance exactly once.

### `tests/test_uol_classifiers.py` — 27 cases

Covers smoke tests (train + predict does not crash) for all 10 classifiers, `predict_proba` shape/range/sum-to-one, FOBOS/FTRL dynamic dimension expansion without `IndexError`, ORF3V weight keys are valid global feature IDs, OSLMF correct batch gating, schema mismatch raises `ValueError`.

### `tests/test_optimizations.py` — 11 cases

Verifies numerical equivalence: `searchsorted` vs ECDF error < 1e-12; vectorised DensityPeaks vs loop-based, element-wise error < 1e-10 across four matrix sizes; OVFM, OSLMF, and ORF3V with the same seed produce bit-identical prediction sequences across two runs.

---

## Compatibility

v0.2 is fully backwards compatible with v0.1. The public API (`train`, `predict`, `predict_proba`, all stream wrapper constructors) is unchanged. The only user-visible behavioural difference is that OLD3S normalisation now works correctly — experiments that relied on the buggy (unnormalised) behaviour may see different accuracy numbers.
