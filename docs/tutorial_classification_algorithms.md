# Classification Algorithms for Evolving Feature Spaces

OpenMOA introduces **10 classification algorithms** specifically designed for Utilitarian Online Learning (UOL) — learning from data streams where the feature space itself changes over time. These algorithms are the core contribution of OpenMOA on top of CapyMOA's existing library of 20+ traditional online classifiers.

This tutorial provides a systematic introduction to each algorithm: its theoretical foundation, how it works, when to use it, and working code examples.

---

## Overview

### Algorithm Taxonomy

The 10 algorithms fall into four categories based on their underlying approach:

| Category | Algorithms | Core Technique |
|---|---|---|
| **Feature Space Mapping** | FESL | Learns a linear mapping between old and new feature spaces |
| **Copula-based Imputation** | OVFM, OSLMF | Uses Gaussian Copula to model missing features statistically |
| **Sparse Online Optimization** | OASF, RSOL, FOBOS, FTRL | Online convex optimization with sparsity-inducing regularization |
| **Deep / Ensemble Models** | OLD3S, ORF3V, OWSS | Neural networks or ensemble methods for evolving spaces |

### Quick Reference

| Algorithm | Multi-class | Sparse-Aware | NaN-Aware | Deep Learning | Scalable to RCV1 |
|---|---|---|---|---|---|
| **FESL** | No (Binary) | Yes | No | No | No (OOM) |
| **OVFM** | No (Binary) | No | Yes | No | No (OOM) |
| **OSLMF** | No (Binary) | No | Yes | No | No (OOM) |
| **OASF** | No (Binary) | Yes | Yes | No | Yes |
| **RSOL** | No (Binary) | Yes | Yes | No | Yes |
| **FOBOS** | Yes | Yes | Yes | No | Yes |
| **FTRL** | Yes | Yes | Yes | No | Yes |
| **OLD3S** | Yes | Yes | No | Yes (PyTorch) | Yes |
| **ORF3V** | Yes | Yes | No | No | Yes |
| **OWSS** | Yes | No | No | Yes (PyTorch) | Moderate |

### Compatibility with Stream Wrappers

| Wrapper | Representation | Compatible Algorithms |
|---|---|---|
| `OpenFeatureStream` | Variable-length vector + `feature_indices` | FESL, OASF, RSOL, FOBOS, FTRL, OLD3S, ORF3V |
| `TrapezoidalStream` | Fixed-length vector with NaN | OVFM, OSLMF, FOBOS, FTRL, OASF, RSOL |
| `CapriciousStream` | Fixed-length vector with NaN | OVFM, OSLMF, FOBOS, FTRL, OASF, RSOL |
| `EvolvableStream` | Fixed-length vector with NaN | OVFM, OSLMF, FOBOS, FTRL, OASF, RSOL |

---

## 1. FESL — Feature Evolvable Streaming Learning

### Background

FESL is the foundational algorithm for learning in evolving feature spaces, proposed at NeurIPS 2017. It addresses the scenario where the entire feature space shifts: the old set of features S_old is gradually replaced by a new set S_new, with a brief overlap period where both are observable.

### How It Works

FESL maintains two linear models and learns a mapping between them:

```
Phase 1 (Stable):  Only S_old features visible → Train w_old
Phase 2 (Overlap): Both S_old and S_new visible → Train w_curr on S_new
                                                  → Learn mapping M: S_new → S_old (Ridge Regression)
Phase 3 (New):     Only S_new features visible  → Ensemble prediction:
                                                    y = μ₁·f_curr(x) + μ₂·f_old(M·x)
```

**Key equations:**

- **Mapping Matrix:** M = (X_new^T X_new + λI)^(-1) X_new^T X_old
- **Ensemble Prediction:** y = μ_curr · σ(w_curr^T x) + μ_old · σ(w_old^T M^T x)
- **Ensemble Weight Update:** μ_k ← μ_k · exp(-η · loss_k), then normalize

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | float | 0.1 | Learning rate for SGD weight updates |
| `lambda_` | float | 0.1 | Ridge regularization strength for mapping matrix M |
| `window_size` | int | 100 | Buffer size B to accumulate overlap instances for learning M |
| `random_seed` | int | 1 | Random seed for reproducibility |

### Usage

```python
from openmoa.datasets import Electricity
from openmoa.stream import OpenFeatureStream
from openmoa.classifier import FESLClassifier
from openmoa.evaluation import ClassificationEvaluator

base_stream = Electricity()
stream = OpenFeatureStream(
    base_stream=base_stream,
    evolution_pattern="eds",
    n_segments=2,
    overlap_ratio=1.0,
    total_instances=10000
)

learner = FESLClassifier(
    schema=base_stream.get_schema(),
    alpha=0.1,
    lambda_=0.1,
    window_size=100
)
evaluator = ClassificationEvaluator(schema=base_stream.get_schema())

while stream.has_more_instances():
    instance = stream.next_instance()
    prediction = learner.predict(instance)
    learner.train(instance)
    evaluator.update(instance.y_index, prediction)

print(f"FESL Accuracy: {evaluator.accuracy():.2f}%")
```

### Complexity

| Aspect | Complexity |
|---|---|
| Per-instance training | O(d) for SGD update |
| Mapping learning | O(B · d² + d³) for Ridge Regression solve |
| Memory | O(d_old × d_new) for mapping matrix M |

### Limitations

- **Binary classification only** (uses logistic regression internally)
- **Memory bottleneck:** The mapping matrix M has size d_old × d_new. For RCV1 (~47k features), this requires ~17 GB — causing OOM on standard machines. This is an inherent algorithmic limitation.
- **Requires overlap period:** Cannot learn the mapping without a period where both old and new features coexist.

### Reference

> Hou, B. J., Zhang, L., & Zhou, Z. H. (2017). Learning with Feature Evolvable Streams. *Advances in Neural Information Processing Systems (NeurIPS)*.

---

## 2. OVFM — Online Learning in Variable Feature Spaces with Mixed Data

### Background

OVFM addresses the challenge of learning when features randomly appear and disappear, and the data contains both continuous and ordinal types. It uses a Gaussian Copula model to statistically impute missing features, then trains an ensemble of two classifiers: one in the observed space and one in the latent (copula-transformed) space.

### How It Works

```
For each mini-batch:
  1. Detect Feature Types → Continuous vs Ordinal (≤14 unique values)
  2. Update Marginal Distributions → Sliding window ECDF per feature
  3. EM Algorithm:
     E-step: Impute missing latent values Z using conditional Gaussian
     M-step: Update correlation matrix Σ
  4. Train Two Classifiers (SGD):
     - w_obs: On observed/reconstructed features
     - w_lat: On latent (copula-transformed) features
  5. Update Ensemble Weight:
     α = exp(-τ·L_obs) / [exp(-τ·L_obs) + exp(-τ·L_lat)]
```

**Key innovation:** The Gaussian Copula decouples marginal distributions from the dependency structure, allowing OVFM to handle mixed data types (continuous + ordinal) and reconstruct missing features through the learned correlation matrix.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_size` | int | 200 | Sliding window for marginal ECDF estimation |
| `batch_size` | int | 50 | Mini-batch size for EM updates |
| `evolution_pattern` | str | "vfs" | Stream pattern: "vfs" (general), "tds", "cds", "eds" |
| `decay_coef` | float | 0.5 | Exponential decay for Sigma update |
| `learning_rate` | float | 0.01 | SGD learning rate |
| `l1_lambda` | float | 0.0 | L1 regularization strength |
| `l2_lambda` | float | 0.01 | L2 regularization strength |
| `max_ord_levels` | int | 14 | Max unique values to classify feature as ordinal |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.datasets import Spambase
from openmoa.stream import TrapezoidalStream
from openmoa.classifier import OVFMClassifier
from openmoa.evaluation import ClassificationEvaluator

base_stream = Spambase()
stream = TrapezoidalStream(
    base_stream=base_stream,
    evolution_mode="random",
    d_min=10,
    total_instances=4601
)

learner = OVFMClassifier(
    schema=base_stream.get_schema(),
    window_size=200,
    batch_size=50,
    learning_rate=0.01
)
evaluator = ClassificationEvaluator(schema=base_stream.get_schema())

while stream.has_more_instances():
    instance = stream.next_instance()
    prediction = learner.predict(instance)
    learner.train(instance)
    evaluator.update(instance.y_index, prediction)

print(f"OVFM Accuracy: {evaluator.accuracy():.2f}%")
```

### Complexity

| Aspect | Complexity |
|---|---|
| EM step (per batch) | O(d³) for conditional Gaussian imputation |
| Memory | O(d²) for correlation matrix Σ |

### Limitations

- **Binary classification only**
- **O(d³) per batch**: Matrix inversion makes it infeasible for very high-dimensional data (RCV1)
- **Requires `statsmodels`**: Uses Empirical CDF from `statsmodels.distributions.empirical_distribution`

### Reference

> He, Y., Dong, J., Hou, B. J., Wang, Y., & Wang, F. (2021). Online Learning in Variable Feature Spaces with Mixed Data. *IEEE International Conference on Data Mining (ICDM)*.

---

## 3. OSLMF — Online Semi-supervised Learning with Mix-Typed Streaming Features

### Background

OSLMF extends the Copula-based approach of OVFM with **semi-supervised learning** capabilities. When labeled data is scarce, it uses Density Peak Clustering to propagate labels from labeled to unlabeled instances in the latent space.

### How It Works

```
For each instance:
  1. Gaussian Copula → Transform to latent space Z
  2. Density Peak Clustering → Propagate labels to unlabeled instances
     - Compute local density ρ and distance to higher density δ
     - Trace nearest higher-density neighbor chain to find label source
  3. Train Two Classifiers (SGD):
     - w_obs: On observed/reconstructed features
     - w_lat: On latent features
  4. Ensemble with adaptive weights
```

**Three components:**
- **Gaussian Copula:** Handles mixed data types and missing features (same as OVFM)
- **Density Peak Clustering:** Label propagation in latent space for semi-supervised learning
- **Dual Classifier Ensemble:** Combines observed-space and latent-space predictions

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_size` | int | 200 | Sliding window for ECDF |
| `buffer_size` | int | 200 | Buffer for density peak clustering |
| `learning_rate` | float | 0.01 | SGD learning rate |
| `decay_coef` | float | 0.5 | Copula covariance decay |
| `l2_lambda` | float | 0.001 | L2 regularization |
| `ensemble_weight` | float | 0.5 | Initial ensemble weight for observed classifier |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.datasets import German
from openmoa.stream import CapriciousStream
from openmoa.classifier import OSLMFClassifier
from openmoa.evaluation import ClassificationEvaluator

base_stream = German()
stream = CapriciousStream(
    base_stream=base_stream,
    missing_ratio=0.3,
    total_instances=1000
)

learner = OSLMFClassifier(
    schema=base_stream.get_schema(),
    window_size=200,
    buffer_size=200,
    learning_rate=0.01
)
evaluator = ClassificationEvaluator(schema=base_stream.get_schema())

while stream.has_more_instances():
    instance = stream.next_instance()
    prediction = learner.predict(instance)
    learner.train(instance)
    evaluator.update(instance.y_index, prediction)

print(f"OSLMF Accuracy: {evaluator.accuracy():.2f}%")
```

### Complexity

| Aspect | Complexity |
|---|---|
| Copula transform | O(d²) |
| Density peaks | O(buffer²) for distance matrix |
| Memory | O(d²) for Σ + O(buffer² ) for distances |

### Limitations

- **Binary classification only**
- **O(d²) memory** for covariance matrix — infeasible for RCV1
- **Requires `statsmodels`**

### Reference

> Wu, D., et al. (2023). Online Semi-supervised Learning with Mix-Typed Streaming Features. *AAAI Conference on Artificial Intelligence*.

---

## 4. OASF — Online Active Sparse Feature Learning

### Background

OASF is designed for streams where features incrementally appear or disappear. It uses Passive-Aggressive (PA) updates with group sparsity regularization via the L1,2-norm, automatically selecting important features over time.

### How It Works

```
For each instance:
  1. Parse sparse input (handles global feature indices)
  2. Auto-expand weight matrix if new features appear
  3. PA Update:
     - Decremental (features disappeared) → Theorem 1: Update on surviving features
     - Incremental (features appeared) → Theorem 2: Pad old weights, update all
     - Loss: max(0, 1 - y·w^T x)
     - γ = loss / (||x||² + 1/2μ)
  4. Slide Window: Shift W columns left, store new weights
  5. L1,2-norm Group Sparsity (Theorem 3):
     - If ||w_row||₂ ≤ λ → zero out entire feature row
     - Else → shrink: w_row ← (1 - λ/||w_row||) · w_row
```

**Key idea:** The sliding window W stores the last L weight vectors. The L1,2-norm operates on rows of W (one row per feature), zeroing out features whose weights have been consistently weak across the window — providing automatic feature selection.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lambda_param` | float | 0.01 | L1,2-norm sparsity regularization |
| `mu` | float | 1.0 | PA smoothness parameter (larger = more conservative updates) |
| `L` | int | 100 | Sliding window size |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.datasets import W8a
from openmoa.stream import OpenFeatureStream, ShuffledStream
from openmoa.classifier import OASFClassifier
from openmoa.evaluation import ClassificationEvaluator

base_stream = W8a()
shuffled = ShuffledStream(base_stream=base_stream, random_seed=42)
stream = OpenFeatureStream(
    base_stream=shuffled,
    evolution_pattern="incremental",
    d_min=50,
    total_instances=10000
)

learner = OASFClassifier(
    schema=base_stream.get_schema(),
    lambda_param=0.01,
    mu=1.0,
    L=100
)
evaluator = ClassificationEvaluator(schema=base_stream.get_schema())

while stream.has_more_instances():
    instance = stream.next_instance()
    prediction = learner.predict(instance)
    learner.train(instance)
    evaluator.update(instance.y_index, prediction)

print(f"OASF Accuracy: {evaluator.accuracy():.2f}%")
print(f"Weight Sparsity: {learner.get_sparsity():.1%}")
```

### Complexity

| Aspect | Complexity |
|---|---|
| Per-instance training | O(d · L) for window update |
| L1,2 sparsity step | O(d · L) vectorized |
| Memory | O(d × L) for weight matrix W |

### Limitations

- **Binary classification only**

### Reference

> Chen, Z., et al. (2024). L1,2-Norm and CUR Decomposition for Online Sparse Feature Learning. *IEEE International Conference on Big Data*.

---

## 5. RSOL — Robust Sparse Online Learning

### Background

RSOL improves upon OASF with two key optimizations: a **ring buffer** for the sliding window (avoiding expensive array shifts) and **auto-expanding weights** for dynamic feature dimensions. It shares the same PA + L1,2-norm theoretical framework.

### How It Works

```
For each instance:
  1. Parse sparse input → auto-expand weight matrix if needed
  2. Retrieve latest weights from ring buffer (pointer-based, O(1))
  3. PA Update:
     - Loss: max(0, 1 - y·w^T x)
     - γ = loss / (||x||² + 1/2μ)
     - w_new = w_prev + γ·y·x
  4. Store in ring buffer at current pointer position
  5. L1,2-norm Sparsity (Theorem 3.3):
     - Row-wise L2 norm soft thresholding
  6. Advance ring buffer pointer: ptr = (ptr + 1) % L
```

**Key improvement over OASF:** The ring buffer replaces `np.roll()` with O(1) pointer advancement, making RSOL significantly faster on large datasets.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lambda_param` | float | 50.0 | L1,2-norm sparsity regularization |
| `mu` | float | 1.0 | PA smoothness parameter |
| `L` | int | 1000 | Window size (ring buffer capacity) |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.classifier import RSOLClassifier

learner = RSOLClassifier(
    schema=stream.get_schema(),
    lambda_param=50.0,
    mu=1.0,
    L=1000
)
```

### Complexity

| Aspect | Complexity |
|---|---|
| Per-instance training | O(d) for PA update + O(d · L) for sparsity |
| Window management | O(1) ring buffer pointer |
| Memory | O(d × L) |

### Limitations

- **Binary classification only**
- Higher default `lambda_param` (50.0) — may need tuning for different datasets

### Reference

> Chen, Z., et al. (2024). Robust Sparse Online Learning. *SIAM International Conference on Data Mining (SDM)*.

---

## 6. FOBOS — Forward-Backward Splitting

### Background

FOBOS (Forward-Backward Splitting) is a general-purpose online learning algorithm from the optimization literature. It separates the gradient descent step (forward) from the regularization step (backward proximal operator), enabling clean L1/L2/Group Lasso sparsity.

**Unlike FESL/OASF/RSOL, FOBOS supports both binary and multi-class classification.**

### How It Works

```
For each instance:
  Step 1 — Forward (Gradient Descent):
    Binary:     w ← w - η·(σ(w^T x) - y)·x      (Logistic Regression)
    Multi-class: W ← W - η·outer(x, softmax(Wx) - e_y)  (Softmax Regression)

  Step 2 — Backward (Proximal Operator):
    L1:    w ← sign(w)·max(0, |w| - ηλ)           (Soft Thresholding)
    L2:    w ← w·(1 - ηλ)                          (Multiplicative Decay)
    L1,2:  w_row ← max(0, 1 - ηλ/||w_row||)·w_row (Group Lasso)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | float | 1.0 | Initial learning rate η |
| `lambda_` | float | 0.001 | Regularization strength |
| `regularization` | str | "l1" | Regularization type: "l1", "l2", "l1_l2", or "none" |
| `step_schedule` | str | "sqrt" | Learning rate decay: "sqrt" (1/sqrt(t)) or "linear" (1/t) |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.classifier import FOBOSClassifier

# Binary classification
learner = FOBOSClassifier(
    schema=stream.get_schema(),
    alpha=1.0,
    lambda_=0.001,
    regularization="l1",
    step_schedule="sqrt"
)

# Multi-class classification (automatically detected from schema)
learner_mc = FOBOSClassifier(
    schema=multiclass_stream.get_schema(),
    alpha=0.5,
    lambda_=0.01,
    regularization="l1_l2"
)
```

### Complexity

| Aspect | Complexity |
|---|---|
| Binary training | O(k) where k = number of active features |
| Multi-class training | O(k · C) where C = number of classes |
| Proximal step | O(d · C) for full weight matrix |
| Memory | O(d × C) |

### Reference

> Duchi, J., & Singer, Y. (2009). Efficient Online and Batch Learning Using Forward Backward Splitting. *Journal of Machine Learning Research*.

---

## 7. FTRL — Follow-The-Regularized-Leader

### Background

FTRL-Proximal is a state-of-the-art online learning algorithm widely used in industry (notably at Google for ad click prediction). It achieves superior sparsity compared to standard SGD by accumulating gradient information and applying L1 thresholding in a principled manner.

**Like FOBOS, FTRL supports both binary and multi-class classification.**

### How It Works

```
FTRL maintains three state vectors per feature:
  z: accumulated gradients minus learning-rate-weighted weights
  n: sum of squared gradients (for adaptive learning rate)
  w: current weights

For each instance:
  1. Compute prediction and gradient g
  2. Update n: n_new = n + g²
  3. Compute σ = (√n_new - √n_old) / α
  4. Update z: z ← z + g - σ·w
  5. Proximal Step (L1 Thresholding):
     if |z_i| ≤ l1:
       w_i = 0                           (Sparsity!)
     else:
       w_i = -(z_i - sign(z_i)·l1) / ((β + √n_i)/α + l2)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | float | 0.1 | Learning rate parameter α |
| `beta` | float | 1.0 | Smoothing parameter β |
| `l1` | float | 1.0 | L1 regularization (controls sparsity) |
| `l2` | float | 1.0 | L2 regularization (controls smoothness) |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.classifier import FTRLClassifier

learner = FTRLClassifier(
    schema=stream.get_schema(),
    alpha=0.1,
    beta=1.0,
    l1=1.0,
    l2=1.0
)

# After training, inspect sparsity
print(f"Weight sparsity: {learner.get_sparsity():.1%}")
```

### Complexity

| Aspect | Complexity |
|---|---|
| Per-instance training | O(k · C) where k = active features |
| Memory | O(3 · d · C) for z, n, w arrays |

### Reference

> McMahan, H. B. (2011). Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization. *International Conference on Artificial Intelligence and Statistics (AISTATS)*.

---

## 8. OLD3S — Online Learning from Data of Double Streams

### Background

OLD3S is a deep learning approach to evolving feature spaces. It uses a **Variational Autoencoder (VAE)** to map each feature space into a shared latent space, and a **Hedge Backpropagation (HBP) MLP** for classification. When the feature space shifts, knowledge from the old model is distilled into the new one via latent-space alignment.

### How It Works

```
Architecture:
  VAE (per feature space): x → [Encoder] → μ, σ → z → [Decoder] → x̂
  HBP-MLP (per feature space): z → [Layer₁ → Exit₁] → [Layer₂ → Exit₂] → ... → [Layer_n → Exit_n]

State Machine:
  STABLE:     Train current VAE + HBP normally
  OVERLAP:    Both old and new features visible
              → Train new VAE with alignment loss: L = L_recon + 0.1·L_KLD + 10·L_align
              → L_align = MSE(z_curr, z_prev)
  STABLE_NEW: Only new features visible
              → Ensemble prediction: logit = w_curr·f_curr(z) + w_prev·f_prev(z)

HBP (Hedge Backpropagation):
  Each MLP layer has an exit classifier with a learned weight.
  Weights decay based on per-exit loss: w_i ← w_i · β^(loss_i)
  Final prediction: weighted sum of all exits.
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `latent_dim` | int | 20 | Dimension of VAE latent space z |
| `hidden_dim` | int | 128 | Hidden layer dimension in VAE encoder/decoder |
| `num_hbp_layers` | int | 3 | Number of layers (exits) in HBP-MLP |
| `learning_rate` | float | 0.001 | Adam optimizer learning rate |
| `beta` | float | 0.99 | HBP weight decay rate |
| `eta` | float | 0.01 | Ensemble weight update rate |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.classifier import OLD3SClassifier

learner = OLD3SClassifier(
    schema=stream.get_schema(),
    latent_dim=20,
    hidden_dim=128,
    num_hbp_layers=3,
    learning_rate=0.001
)
```

### Complexity

| Aspect | Complexity |
|---|---|
| Per-instance training | O(d · h + h · z + z · C) for forward/backward pass |
| Memory | O(d · h + h · z) for VAE parameters |

### Limitations

- **Requires PyTorch**
- Computationally heavier than linear methods due to neural network training
- Automatic feature space shift detection relies on `feature_indices` metadata

### Reference

> Lian, H., et al. (2024). Online Learning Deep Models from Data of Double Streams. *IEEE Transactions on Knowledge and Data Engineering (TKDE)*.

---

## 9. ORF3V — Online Random Feature Forests for Varying Feature Spaces

### Background

ORF3V takes a unique approach: instead of maintaining a single global model, it builds **independent decision stump ensembles ("feature forests") for each individual feature**. This naturally handles varying feature spaces — when a feature is absent, its forest is simply excluded from the vote.

### How It Works

```
Architecture:
  Feature Forest[i] = {Stump₁, Stump₂, ..., Stump_k}  (one forest per feature)

For each instance:
  1. Training Phase (after grace_period):
     - Update per-feature statistics (min, max, class-conditional samples)
     - Periodically replace weakest stumps with newly generated ones
     - Update feature weights based on prediction correctness

  2. Prediction Phase:
     For each active feature i with value v:
       - Each stump splits on a threshold: v < split → class_dist_below, else → class_dist_above
       - Aggregate all stumps in forest[i] weighted by stump weights
     Final: class_scores[c] = Σ weight[i] · forest[i].predict(v)[c]

  3. Pruning (Optional, Hoeffding Bound):
     If a feature's availability drops below Hoeffding bound → prune its forest
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_stumps` | int | 10 | Number of decision stumps per feature forest |
| `alpha` | float | 0.1 | Learning rate for weight updates |
| `grace_period` | int | 100 | Instances before initializing forests |
| `replacement_interval` | int | 100 | How often to replace stumps |
| `replacement_strategy` | str | "oldest" | Stump replacement: "oldest" or "random" |
| `window_size` | int | 200 | Sliding window size for pruning |
| `delta` | float | 0.001 | Hoeffding bound confidence parameter |
| `enable_pruning` | bool | False | Enable Hoeffding-bound feature pruning |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.classifier import ORF3VClassifier

learner = ORF3VClassifier(
    schema=stream.get_schema(),
    n_stumps=10,
    alpha=0.1,
    grace_period=100,
    replacement_interval=100
)
```

### Complexity

| Aspect | Complexity |
|---|---|
| Per-instance training | O(d · n_stumps) |
| Per-instance prediction | O(k · n_stumps) where k = active features |
| Memory | O(d · n_stumps · samples) |

### Key Advantage

ORF3V **does not require any imputation or mapping** — missing features are simply excluded. This makes it naturally robust to arbitrary patterns of feature appearance and disappearance.

### Reference

> Schreckenberger, C., He, Y., Ludtke, S., Bartelt, C., & Stuckenschmidt, H. (2023). Online Random Feature Forests for Learning in Varying Feature Spaces. *AAAI Conference on Artificial Intelligence*.

---

## 10. OWSS — Utilitarian Online Learning from Open-World Soft Sensing

### Background

OWSS uses a **Graph Neural Network (GNN)** approach with a bipartite graph structure connecting instances to features. It learns universal feature embeddings that persist across changing feature spaces, enabling knowledge transfer when features appear or disappear.

### How It Works

```
Architecture:
  Feature Embeddings: f₁, f₂, ..., f_d  (learned, persistent)
  Input Projector: Maps raw scalar value → hidden dimension
  Bipartite Graph: Connects instances to their active features
  GCN Layer: Message passing on the bipartite graph
  Classifier Head: MLP on GCN-refined instance representations

For each mini-batch:
  1. Build Bipartite Adjacency Matrix:
     Nodes = [Feature₁...Feature_d | Instance₁...Instance_B]
     Edges: Instance_i ↔ Feature_j if x_{ij} > threshold
  2. Initial Instance Representation: z_t = X · FeatureEmbeddings
  3. GCN Message Passing: Refine representations
  4. Losses:
     - L_cls: Cross-entropy classification loss
     - L_rec: Feature reconstruction loss MSE(z_GCN, z_initial)
     - Total: L = L_cls + β · L_rec
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_size` | int | 100 | Mini-batch size |
| `hidden_dim` | int | 32 | GCN hidden dimension |
| `learning_rate` | float | 0.01 | Adam optimizer learning rate |
| `rec_weight` | float | 0.1 | Weight β for reconstruction loss |
| `sparsity_threshold` | float | 0.05 | Edge pruning threshold for dense data |
| `random_seed` | int | 1 | Random seed |

### Usage

```python
from openmoa.classifier import OWSSClassifier

learner = OWSSClassifier(
    schema=stream.get_schema(),
    window_size=100,
    hidden_dim=32,
    learning_rate=0.01,
    rec_weight=0.1
)
```

### Complexity

| Aspect | Complexity |
|---|---|
| Per-batch training | O(B · d · h) for graph construction + GCN |
| Memory | O(d · h + (B + d)²) for embeddings + adjacency |

### Limitations

- **Requires PyTorch**
- **Batch-based**: Updates happen when buffer reaches `window_size`
- The bipartite adjacency matrix can be large for high-dimensional data

### Reference

> Lian, H., et al. (2024). Utilitarian Online Learning from Open-World Soft Sensing. *IEEE International Conference on Data Mining (ICDM)*.

---

## Comparing Algorithms: A Decision Guide

### By Dataset Size and Dimensionality

| Scenario | Recommended Algorithms |
|---|---|
| **Low-dimensional** (d < 100, e.g., German, Australian) | OVFM, OSLMF, OLD3S, ORF3V |
| **Medium-dimensional** (d = 100–1000, e.g., W8a, Musk) | FESL, OASF, FOBOS, FTRL, ORF3V |
| **High-dimensional** (d > 1000, e.g., InternetAds) | OASF, RSOL, FOBOS, FTRL |
| **Very high-dimensional sparse** (d ~ 47k, e.g., RCV1) | RSOL, FOBOS, FTRL |

### By Feature Evolution Pattern

| Evolution Pattern | Best Algorithms | Why |
|---|---|---|
| **EDS** (Feature space shifts with overlap) | FESL, OLD3S | Designed for explicit S_old → S_new transitions |
| **TDS** (Features gradually appear) | OVFM, OASF, FOBOS, FTRL | Handle monotonically growing feature space |
| **CDS** (Random missing features) | OVFM, OSLMF, ORF3V | Imputation (OVFM) or per-feature independence (ORF3V) |
| **General / Mixed** | FOBOS, FTRL, RSOL | Robust sparse methods that work across all patterns |

### By Classification Task

| Task | Available Algorithms |
|---|---|
| **Binary classification** | All 10 algorithms |
| **Multi-class classification** | FOBOS, FTRL, OLD3S, ORF3V, OWSS |

---

## Complete Benchmark Example

Here is a complete example comparing multiple algorithms on the same evolving feature stream:

```python
from openmoa.datasets import Electricity
from openmoa.stream import OpenFeatureStream, ShuffledStream
from openmoa.classifier import (
    FESLClassifier, OASFClassifier, RSOLClassifier,
    FOBOSClassifier, FTRLClassifier
)
from openmoa.evaluation import ClassificationEvaluator

# Setup stream
base = Electricity()
schema = base.get_schema()

algorithms = {
    "FESL": FESLClassifier(schema=schema, alpha=0.1),
    "OASF": OASFClassifier(schema=schema, lambda_param=0.01),
    "RSOL": RSOLClassifier(schema=schema, lambda_param=50.0),
    "FOBOS": FOBOSClassifier(schema=schema, alpha=1.0, regularization="l1"),
    "FTRL": FTRLClassifier(schema=schema, alpha=0.1, l1=1.0, l2=1.0),
}

for name, learner in algorithms.items():
    base.restart()
    stream = OpenFeatureStream(
        base_stream=base,
        evolution_pattern="pyramid",
        d_min=2,
        total_instances=10000
    )
    evaluator = ClassificationEvaluator(schema=schema)

    while stream.has_more_instances():
        instance = stream.next_instance()
        prediction = learner.predict(instance)
        learner.train(instance)
        evaluator.update(instance.y_index, prediction)

    print(f"{name:8s} → Accuracy: {evaluator.accuracy():.2f}%")
```

---

## Next Steps

- **Concept Drift Detection** — Learn how to detect and respond to distribution changes
- **Evaluation Methods** — Master prequential evaluation with windowed metrics and multi-learner comparison
- **Advanced Topics** — Explore regression variants (FESLRegressor, OASFRegressor), anomaly detection, and AutoML
