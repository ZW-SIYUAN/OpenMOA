# OpenMOA v0.1 — Introducing Utilitarian Online Learning for Dynamic Feature Spaces

OpenMOA is a Python library for **Utilitarian Online Learning (UOL)** — a paradigm designed for real-world streaming environments where the feature space itself evolves over time. Built on top of [CapyMOA](https://capymoa.org), OpenMOA provides a clean, unified API that integrates MOA online learners, CapyMOA's stream learning backend, and PyTorch deep models.

---

## Why OpenMOA?

Most online learning libraries assume a **fixed feature space**: every instance arrives with the same set of features, in the same order, with the same semantics. Real-world data streams routinely violate this assumption.

Sensors go offline. New signals are added mid-deployment. Features phase in or out according to external schedules. In these settings, standard online learners silently produce wrong predictions — or crash outright.

OpenMOA addresses this gap with two purpose-built layers:

- **Stream Wrappers** that simulate diverse feature-space dynamics on top of any static dataset
- **UOL Classifiers** that are designed from the ground up to handle evolving feature spaces

---

## Installation

OpenMOA requires Java (for MOA) and PyTorch.

```bash
# Check Java
java -version

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install OpenMOA
pip install openmoa

# Verify
python -c "import openmoa; print(openmoa.__version__)"
```

For GPU support or troubleshooting, see the [installation guide](https://openmoa.net/docs/getting-started/).

---

## Core Components

### Stream Wrappers

OpenMOA wraps any static dataset and transforms it into a dynamic stream that simulates feature-space evolution. Five wrapper types cover the major paradigms studied in the literature.

#### `OpenFeatureStream` — Variable-Length Output with Global Index Mapping

The flagship wrapper. At each time step, it exposes only the currently *active* features and attaches a `feature_indices` attribute to every instance — a list of global feature IDs corresponding to the values in `instance.x`. Downstream learners use this to maintain consistent weight indexing across changing feature sets.

Supports six evolution modes: EDS (Evolvable), TDS (Trapezoidal), CDS (Capricious), random, cyclic, and custom schedules.

```python
from openmoa.stream import OpenFeatureStream
from capymoa.datasets import Electricity

stream = OpenFeatureStream(
    base_stream=Electricity(),
    mode="EDS",
    n_segments=5,
    overlap_ratio=0.5,
)

instance = stream.next_instance()
print(instance.feature_indices)  # e.g. [0, 2, 5] — active global feature IDs
print(instance.x)                # values for those three features only
```

#### `TrapezoidalStream` — Features Phase In Gradually (TDS)

Implements the Trapezoidal Data Stream paradigm (Gao et al.). Features enter the stream in stages: each feature has a birth stage and a retirement stage, creating a trapezoidal activation window. Inactive features are represented as `NaN` in a fixed-width vector, preserving compatibility with learners that expect constant input dimensionality.

#### `CapriciousStream` — Random Missing Features (CDS)

At each time step, features are independently masked with a configurable `missing_ratio`. Missing values appear as `NaN`. Models must tolerate sparse, unpredictably incomplete input.

```python
from openmoa.stream import CapriciousStream

stream = CapriciousStream(base_stream=..., missing_ratio=0.3)
# ~30% of feature values will be NaN at each step
```

#### `EvolvableStream` — Segmented Feature Rotation (EDS)

Divides the stream into alternating *stable* and *transition* segments. During stable segments, a fixed feature subset is active. During transitions, old features fade out while new ones phase in. Output is fixed-width with `NaN` fill.

#### `ShuffledStream` — Eliminates Label-Sorting Bias

Many benchmark datasets are sorted by class label, which artificially inflates the performance of any learner with a memory component. `ShuffledStream` buffers the entire dataset in memory and shuffles instances before replay, ensuring evaluation reflects truly random arrival order.

---

### UOL Classifiers

Ten algorithms, all implementing a common interface (`train(instance)` / `predict(instance)` / `predict_proba(instance)`), all compatible with `OpenFeatureStream`'s variable-length input.

| Algorithm | Description |
|-----------|-------------|
| **FESL** | Feature-Evolving Sparse Learning. The core UOL algorithm: maintains a sparse weight vector in a global feature ID space and handles feature-space transitions via explicit mapping matrices. |
| **OASF** | Online Adaptive Sparse Filtering. Sliding-window weight matrix; adapts to changing feature relevance over time. |
| **RSOL** | Randomly Sparse Online Learning. Random projection-based sparse updates with provable regret bounds. |
| **FOBOS** | Forward-Backward Splitting. Proximal gradient descent with L1 regularization for online sparse learning. |
| **FTRL** | Follow The Regularized Leader. Per-coordinate adaptive learning rates with L1+L2 regularization. |
| **ORF3V** | Online Random Forest for Variable-length feature Vectors. Extends random forests to streams with dynamic feature spaces by maintaining per-feature-ID tree structures. |
| **OVFM** | Online Variational Factorization Machine. Models feature interactions via a Gaussian Copula; designed for mixed continuous/ordinal feature types. |
| **OSLMF** | Online Semi-supervised Local Matrix Factorization. Combines DensityPeaks clustering with a Copula-based latent representation for label-efficient learning. |
| **OLD3S** | Online Deep Dynamic Semi-Supervised Streams. VAE-based deep model with hierarchical Bayesian pruning (HBP); supports unlabeled instances. |
| **OWSS** | Online Weighted Sparse Streams. Importance-weighted sparse updates for non-stationary distributions. |

All classifiers accept instances from any OpenMOA stream wrapper, including the variable-length output of `OpenFeatureStream`.

```python
from openmoa.stream import OpenFeatureStream
from openmoa.classifier import FESLClassifier
from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation

stream = OpenFeatureStream(Electricity(), mode="EDS", n_segments=5)
learner = FESLClassifier(schema=stream.get_schema())

results = prequential_evaluation(stream, learner, window_size=500)
print(results.accuracy())
```

---

## Datasets

OpenMOA v0.1 ships with benchmark support for **17 datasets** — 11 binary classification and 6 multi-class — drawn from UCI, MOA's built-in repository, and domain-specific streaming benchmarks.

All 17 datasets are evaluated under all three feature-space paradigms (TDS, CDS, EDS), yielding a comprehensive 10-algorithm × 17-dataset × 3-paradigm benchmark grid.

Benchmark code: [`demo/demo_fesl_benchmark_binary.py`](https://github.com/ZW-SIYUAN/OpenMOA/blob/main/demo/demo_fesl_benchmark_binary.py)

---

## Evaluation Protocol

All experiments follow the **prequential (test-then-train)** protocol: each instance is first used for evaluation, then for training. This produces unbiased estimates of online accuracy without a held-out test set. Prequential evaluation is the standard protocol in stream learning research and is natively supported by CapyMOA's evaluation utilities.

---

## Cite

If you use OpenMOA in your research, please cite:

```bibtex
@misc{ZhiliWang2025OpenMOA,
    title={{OpenMOA}: A Python Library for Utilitarian Online Learning},
    author={Zhili Wang, Heitor M. Gomes and Yi He},
    year={2025},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/},
}
```

---

## Links

- [Documentation](https://openmoa.net/docs/)
- [GitHub](https://github.com/ZW-SIYUAN/OpenMOA)
- [PyPI](https://pypi.org/project/openmoa/)
- [Discord](https://discord.gg/spd2gQJGAb)
