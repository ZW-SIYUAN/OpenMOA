# Working with Data Streams

In traditional machine learning, we assume access to a complete, static dataset. In streaming machine learning, data arrives **one instance at a time** in a potentially infinite sequence. The model must learn incrementally — making predictions and updating itself on the fly — without ever storing the entire dataset in memory.

OpenMOA provides a rich, unified `Stream` API for creating, transforming, and consuming data streams from a variety of sources: built-in benchmark datasets, local files (ARFF, CSV, LIBSVM), NumPy arrays, PyTorch datasets, synthetic generators, and specialized stream wrappers for simulating **feature space evolution**.

---

## 1. Core Concepts

### 1.1 What Is a Data Stream?

A data stream is an ordered sequence of instances that arrive over time. Each instance consists of:

- **Feature vector `x`**: An array of attribute values (numeric or categorical).
- **Label `y`**: A class index (for classification) or a numeric value (for regression).

In OpenMOA, every data stream inherits from the abstract base class `Stream` and implements three essential methods:

| Method | Description |
|---|---|
| `next_instance()` | Returns the next instance in the stream |
| `has_more_instances()` | Returns `True` if more instances are available |
| `get_schema()` | Returns a `Schema` object describing the feature space |

### 1.2 The Schema Object

A `Schema` describes the structure of the data:

- Number and names of attributes (features)
- Number and names of classes (for classification)
- Whether the task is classification or regression

```python
from openmoa.datasets import Electricity

stream = Electricity()
schema = stream.get_schema()

print(f"Number of attributes: {schema.get_num_attributes()}")
print(f"Number of classes: {schema.get_num_classes()}")
print(f"Is classification: {schema.is_classification()}")
print(f"Is regression: {schema.is_regression()}")
```

### 1.3 The Instance Object

When you call `next_instance()`, you receive either:

- A **`LabeledInstance`** (for classification) with attributes `x` (numpy array), `y_index` (integer class index), and `y_label` (string class name).
- A **`RegressionInstance`** (for regression) with attributes `x` (numpy array) and `y_value` (float target value).

```python
instance = stream.next_instance()

print(f"Features: {instance.x}")
print(f"Class index: {instance.y_index}")
print(f"Class label: {instance.y_label}")
```

### 1.4 The Test-Then-Train Loop

The fundamental pattern in stream learning is the **test-then-train** (prequential) loop. For each incoming instance, you:

1. **Predict** first (test on unseen data)
2. **Train** with the true label (update the model)
3. **Evaluate** the prediction against the true label

> **Important:** You must never train before testing, as this would leak future information and produce unreliable evaluations.

```python
from openmoa.datasets import Electricity
from openmoa.classifier import HoeffdingTree
from openmoa.evaluation import ClassificationEvaluator

stream = Electricity()
learner = HoeffdingTree(schema=stream.get_schema())
evaluator = ClassificationEvaluator(schema=stream.get_schema())

while stream.has_more_instances():
    instance = stream.next_instance()

    prediction = learner.predict(instance)   # Step 1: Test
    learner.train(instance)                  # Step 2: Train
    evaluator.update(instance.y_index, prediction)  # Step 3: Evaluate

print(f"Accuracy: {evaluator.accuracy():.2f}%")
```

Alternatively, you can use the Pythonic iterator pattern:

```python
for instance in stream:
    prediction = learner.predict(instance)
    learner.train(instance)
    evaluator.update(instance.y_index, prediction)
```

---

## 2. Loading Data Streams

OpenMOA supports multiple ways to create data streams. This section covers each method with working code examples.

### 2.1 Built-in Datasets

OpenMOA ships with a collection of widely-used benchmark datasets. Simply import the dataset class and instantiate it — the data will be automatically downloaded if not already present on disk.

**Classification Datasets:**

| Dataset | Instances | Attributes | Classes | Description |
|---|---|---|---|---|
| `Electricity` | 45,312 | 8 | 2 | Australian electricity price changes |
| `Covtype` | 581,012 | 54 | 7 | Forest cover type prediction |
| `Sensor` | 2,219,803 | 5 | 54 | Intel Lab indoor sensor data |
| `Hyper100k` | 100,000 | 10 | 2 | Moving hyperplane (synthetic, with drift) |
| `RBFm_100k` | 100,000 | 10 | 5 | Radial Basis Function (synthetic, with drift) |
| `RTG_2abrupt` | 100,000 | 30 | 5 | Random Tree with 2 abrupt drifts |

**Regression Datasets:**

| Dataset | Instances | Attributes | Description |
|---|---|---|---|
| `Fried` | 40,768 | 10 | Friedman function-based regression |
| `Bike` | - | - | Bike sharing demand prediction |

**UOL Benchmark Datasets (Binary Classification):**

These datasets are specifically used for benchmarking Utilitarian Online Learning algorithms under evolving feature spaces:

| Dataset | Instances | Attributes | Description |
|---|---|---|---|
| `RCV1` | 20,242 | ~47,000 | Reuters Corpus (sparse, high-dimensional) |
| `W8a` | 49,749 | 300 | Web data classification |
| `Adult` | 32,561 | 123 | Census income prediction |
| `Magic04` | 19,020 | 10 | Gamma telescope signal detection |
| `Spambase` | 4,601 | 57 | Email spam detection |
| `Musk` | 6,598 | 166 | Molecule shape classification |
| `InternetAds` | 2,359 | 1,558 | Online ad detection |
| `German` | 1,000 | 24 | Credit risk assessment |
| `Australian` | 690 | 14 | Credit approval |
| `Ionosphere` | 351 | 34 | Radar signal classification |
| `SVMGuide3` | 1,243 | 21 | LIBSVM benchmark |

**UOL Benchmark Datasets (Multi-Class Classification):**

| Dataset | Instances | Attributes | Classes | Description |
|---|---|---|---|---|
| `DryBean` | 13,611 | 16 | 7 | Dry bean species classification |
| `Optdigits` | 5,620 | 64 | 10 | Handwritten digit recognition |
| `Frogs` | 7,195 | 22 | 4 | Frog species by call features |
| `Wine` | 178 | 13 | 3 | Wine cultivar classification |
| `Splice` | 3,190 | 60 | 3 | DNA splice junction classification |

```python
from openmoa.datasets import Electricity, Fried, Magic04, RCV1

# Classification stream
elec_stream = Electricity()
print(f"Electricity: {len(elec_stream)} instances, {elec_stream.get_schema().get_num_attributes()} features")

# Regression stream
fried_stream = Fried()

# UOL benchmark dataset (binary classification)
magic_stream = Magic04()

# High-dimensional sparse dataset
rcv1_stream = RCV1()
```

### 2.2 Streams from Files

#### ARFF Files

The Attribute-Relation File Format (ARFF) is the native format for MOA and Weka.

```python
from openmoa.stream import ARFFStream

stream = ARFFStream(path="path/to/dataset.arff")
```

#### CSV Files

```python
from openmoa.stream import CSVStream

stream = CSVStream(
    csv_file_path="path/to/data.csv",
    class_index=-1,           # Column index of the target (default: last column)
    delimiter=",",
    dataset_name="MyDataset"
)
```

#### LIBSVM Files

For sparse datasets stored in LIBSVM format (commonly used for large-scale datasets like RCV1):

```python
from openmoa.stream import LibsvmStream

stream = LibsvmStream(path="path/to/data.libsvm")
```

#### Generic File Loading

OpenMOA also provides a convenience function that auto-detects the format:

```python
from openmoa.stream import stream_from_file

stream = stream_from_file(path_to_csv_or_arff="path/to/dataset.arff")
```

### 2.3 Streams from NumPy Arrays

If your data is already loaded in Python as NumPy arrays, you can wrap it into a stream directly:

```python
import numpy as np
from openmoa.stream import NumpyStream

X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])
y = np.array([0, 1, 0])

stream = NumpyStream(X, y, dataset_name="MyDataset")

for instance in stream:
    print(f"x={instance.x}, y={instance.y_index}")
```

For regression tasks, simply pass numeric target values:

```python
y_reg = np.array([1.5, 3.7, 2.1])
stream = NumpyStream(X, y_reg, dataset_name="RegressionData", target_type="numeric")
```

### 2.4 Streams from PyTorch Datasets

OpenMOA seamlessly integrates with PyTorch datasets:

```python
import torch
from torch.utils.data import TensorDataset
from openmoa.stream import TorchClassifyStream

X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = torch.tensor([0, 1, 2])

dataset = TensorDataset(X, y)
stream = TorchClassifyStream(
    dataset=dataset,
    num_classes=3,
    shuffle=True,        # Randomly sample instances
    shuffle_seed=42
)
```

---

## 3. Synthetic Stream Generators

For controlled experiments, OpenMOA provides synthetic data generators through the MOA backend. These generators produce **infinite** streams with known properties, making them ideal for studying algorithm behavior under specific conditions.

### 3.1 Available Generators

| Generator | Description | Attributes | Classes |
|---|---|---|---|
| `RandomTreeGenerator` | Generates a random decision tree; instances are classified by tree structure | Configurable | Configurable |
| `SEA` | SEA concepts with 3 features, used for concept drift studies | 3 | 2 |
| `HyperplaneGenerator` | Rotating hyperplane for gradual drift simulation | Configurable | 2 |
| `AgrawalGenerator` | Simulates a loan application scenario | 9 | 2 |
| `RandomRBFGenerator` | Radial Basis Function centroids for clustering-style classification | Configurable | Configurable |
| `RandomRBFGeneratorDrift` | RBF with drifting centroids | Configurable | Configurable |
| `LEDGenerator` | LED display digit recognition | 7 (or 24 with noise) | 10 |
| `WaveformGenerator` | Waveform classification | 21 (or 40 with noise) | 3 |
| `STAGGERGenerator` | Classic STAGGER concepts | 3 | 2 |
| `SineGenerator` | Sine-based classification boundary | 2 | 2 |
| `HyperplaneGeneratorForRegression` | Regression version of the hyperplane generator | Configurable | - |

### 3.2 Basic Usage

```python
from openmoa.stream.generator import RandomTreeGenerator

stream = RandomTreeGenerator(
    instance_random_seed=1,
    tree_random_seed=1,
    num_classes=2,
    num_nominals=5,
    num_numerics=5,
    max_tree_depth=5
)

# Generators produce infinite streams — use max_instances to limit
instance = stream.next_instance()
print(f"Features: {instance.x}")
print(f"Label: {instance.y_label}")
```

### 3.3 Using Generators with Evaluation

Since synthetic generators produce infinite streams, you must specify `max_instances` when using them with evaluation functions:

```python
from openmoa.stream.generator import SEA
from openmoa.classifier import HoeffdingTree
from openmoa.evaluation import prequential_evaluation

stream = SEA(function=1)
learner = HoeffdingTree(schema=stream.get_schema())

results = prequential_evaluation(
    stream=stream,
    learner=learner,
    window_size=1000,
    max_instances=10000
)

print(f"Accuracy: {results.cumulative.accuracy():.2f}%")
```

---

## 4. Concept Drift Streams

One of the most challenging aspects of data streams is **concept drift** — when the underlying data distribution changes over time. OpenMOA provides the `DriftStream` API for composing complex drifting scenarios from simple building blocks.

### 4.1 Types of Concept Drift

| Type | Description | Width |
|---|---|---|
| **Abrupt** | Instantaneous change from one concept to another | width = 1 (default) |
| **Gradual** | Smooth transition between two concepts over a period | width > 1 |
| **Recurring** | A previously seen concept reappears later in the stream | Configurable |

### 4.2 Building a DriftStream

The DriftStream API uses a list-based syntax where you alternate between **concepts** (stream generators) and **drift events**:

```python
from openmoa.stream.generator import SEA
from openmoa.stream.drift import DriftStream, AbruptDrift, GradualDrift

# Compose a stream with:
#   - SEA function 1 → Abrupt drift at t=5000 → SEA function 3 → Gradual drift → SEA function 1
stream = DriftStream(stream=[
    SEA(function=1),
    AbruptDrift(position=5000),
    SEA(function=3),
    GradualDrift(position=10000, width=2000),  # or: GradualDrift(start=9000, end=11000)
    SEA(function=1),
])
```

### 4.3 Visualizing Drift Effects

The `DriftStream` object carries drift metadata, which is automatically used by OpenMOA's visualization functions to annotate plots:

```python
from openmoa.classifier import OnlineBagging
from openmoa.evaluation import prequential_evaluation
from openmoa.evaluation.visualization import plot_windowed_results

learner = OnlineBagging(schema=stream.get_schema(), ensemble_size=10)

results = prequential_evaluation(
    stream=stream,
    learner=learner,
    window_size=100,
    max_instances=15000
)

# Drift locations are automatically shown as vertical lines on the plot
plot_windowed_results(results, metric="accuracy")
```

### 4.4 Recurring Concept Drift

OpenMOA provides a dedicated API for recurring concepts, where the same concept may appear multiple times in the stream:

```python
from openmoa.stream.drift import RecurrentConceptDriftStream, AbruptDrift
from openmoa.stream.generator import RandomTreeGenerator

concept1 = RandomTreeGenerator(tree_random_seed=1)
concept2 = RandomTreeGenerator(tree_random_seed=2)
concept3 = RandomTreeGenerator(tree_random_seed=3)

stream = RecurrentConceptDriftStream(
    concept_list=[concept1, concept2, concept3],
    max_recurrences_per_concept=2,
    transition_type_template=AbruptDrift(position=2000),
    concept_name_list=["Concept A", "Concept B", "Concept C"]
)
```

---

## 5. Stream Composition Utilities

### 5.1 ConcatStream — Chaining Multiple Streams

`ConcatStream` joins multiple streams end-to-end. When the first stream is exhausted, it seamlessly moves to the next:

```python
from openmoa.stream import ConcatStream, NumpyStream
import numpy as np

stream1 = NumpyStream(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
stream2 = NumpyStream(np.array([[5, 6], [7, 8]]), np.array([1, 0]))

combined = ConcatStream([stream1, stream2])

for instance in combined:
    print(instance.x, instance.y_index)
# Output: [1. 2.] 0, [3. 4.] 1, [5. 6.] 1, [7. 8.] 0
```

> **Note:** All streams in a `ConcatStream` must share the same schema (same number of features and classes).

### 5.2 ShuffledStream — Randomizing Instance Order

When using static datasets (e.g., UCI datasets like Magic04 or Spambase) for online learning experiments, the original file order may be sorted by label, which creates an unrealistic streaming scenario. `ShuffledStream` solves this by buffering all instances and presenting them in random order:

```python
from openmoa.stream import ShuffledStream
from openmoa.datasets import Magic04

base_stream = Magic04()

# Shuffle instance order with a fixed seed for reproducibility
shuffled = ShuffledStream(base_stream=base_stream, random_seed=42)
```

> **Warning:** `ShuffledStream` loads all instances into memory. It is safe for UCI-scale datasets (thousands to tens of thousands of instances), but should NOT be used for massive streams (millions of instances).

**When to use ShuffledStream:**
- Your dataset is a static file (CSV, ARFF) that might be sorted by class label
- You want to simulate a realistic i.i.d. streaming scenario from a batch dataset
- The dataset fits comfortably in memory

**When NOT to use ShuffledStream:**
- Your data has inherent temporal ordering (e.g., Electricity, Sensor)
- The data is generated from a synthetic generator (already random)
- The dataset is too large for memory

### 5.3 Restarting Streams

All OpenMOA streams support the `restart()` method, which resets the stream to read from the beginning. This is useful for running multiple experiments on the same data:

```python
stream = Electricity()

# First run
for instance in stream:
    pass

# Reset and run again
stream.restart()
for instance in stream:
    pass
```

> **Note:** The `prequential_evaluation()` function automatically calls `restart()` by default, so you don't need to manually restart streams between evaluations.

---

## 6. Feature-Evolving Streams (OpenMOA Exclusive)

This is the **core innovation** of OpenMOA. In real-world applications, the feature space itself can change over time — new sensors come online, old ones fail, data sources are added or removed. This is fundamentally different from concept drift, where the feature space stays fixed but the relationship between features and labels changes.

OpenMOA provides **five specialized stream wrappers** that transform any fixed-feature stream into a stream with evolving features. These wrappers implement different evolution paradigms from the research literature on Utilitarian Online Learning.

### 6.1 Overview: Two Representation Strategies

OpenMOA supports two fundamentally different ways to represent instances with missing features:

| Strategy | Wrapper Classes | Representation | Vector Size | Best For |
|---|---|---|---|---|
| **Sparse-Aware** | `OpenFeatureStream` | Subsetted feature vector + global index metadata | Variable (changes over time) | Algorithms that can handle index-based sparse input (FESL, OASF, RSOL) |
| **NaN-Padded** | `TrapezoidalStream`, `CapriciousStream`, `EvolvableStream` | Fixed-size vector with `NaN` for missing features | Fixed (`d_max`) | Algorithms that handle missing values natively (OVFM) |

### 6.2 OpenFeatureStream — The Universal Evolving Stream Wrapper

`OpenFeatureStream` is the most versatile wrapper. It takes any base stream and simulates feature evolution by selecting a subset of features at each time step. The key innovation is attaching `feature_indices` metadata to each instance, which tells downstream algorithms the **global IDs** of the currently active features.

**Supported Evolution Patterns:**

#### Pyramid

Features first increase from `d_min` to `d_max` (first half), then decrease back to `d_min` (second half).

```
Time:     0 ────────── T/2 ────────── T
Features: d_min ──→ d_max ──→ d_min
```

```python
from openmoa.stream import OpenFeatureStream
from openmoa.datasets import Electricity

base = Electricity()
stream = OpenFeatureStream(
    base_stream=base,
    evolution_pattern="pyramid",
    d_min=2,
    d_max=8,
    total_instances=10000,
    feature_selection="prefix"  # "prefix", "suffix", or "random"
)
```

#### Incremental

Features monotonically increase from `d_min` to `d_max` over the entire stream.

```
Time:     0 ──────────────────────── T
Features: d_min ────────────→ d_max
```

```python
stream = OpenFeatureStream(
    base_stream=base,
    evolution_pattern="incremental",
    d_min=2,
    d_max=8,
    total_instances=10000
)
```

#### Decremental

Features monotonically decrease from `d_max` to `d_min`.

```
Time:     0 ──────────────────────── T
Features: d_max ────────────→ d_min
```

```python
stream = OpenFeatureStream(
    base_stream=base,
    evolution_pattern="decremental",
    d_min=2,
    d_max=8,
    total_instances=10000
)
```

#### TDS (Trapezoidal Data Stream)

Each feature has a distinct "birth time." Once a feature appears, it remains available for the rest of the stream. Supports two modes:

- **`ordered`**: Feature 0 appears first, then Feature 1, etc.
- **`random`**: Features appear in a random order.

```
Time:     0 ──── t1 ──── t2 ──── t3 ────── T
Feature0: ████████████████████████████████████
Feature1:        █████████████████████████████
Feature2:               ████████████████████
Feature3:                      █████████████
```

```python
stream = OpenFeatureStream(
    base_stream=base,
    evolution_pattern="tds",
    tds_mode="random",   # or "ordered"
    total_instances=10000
)
```

**Reference:** Zhang, P., Zhu, X., & Guo, L. (2010). Mining Data Streams with Labeled and Unlabeled Training Examples. *ICDM*.

#### CDS (Capricious Data Stream)

At each time step, every feature independently undergoes a Bernoulli trial. If it "fails," the feature is missing for that instance. This produces a stochastic, per-instance pattern of missing features.

```python
stream = OpenFeatureStream(
    base_stream=base,
    evolution_pattern="cds",
    missing_ratio=0.3,    # 30% probability of each feature being missing
    total_instances=10000
)
```

**Reference:** He, Y., Wu, B., Liu, D., et al. (2019). Streaming Feature Selection for Multi-Label Learning based on Fuzzy Mutual Information. *ICFNDS*.

#### EDS (Evolvable Data Stream)

The feature space evolves in **n sequential segments** with overlapping transition periods. The stream is divided into `2n-1` stages:

- **Stable stages**: Only features from one segment are active
- **Overlap stages**: Features from two adjacent segments are both active

```
Segment 1:  ██████████████████
Overlap:            ██████████████████
Segment 2:                  ██████████████████
Overlap:                          ██████████████████
Segment 3:                                ██████████████████
```

```python
stream = OpenFeatureStream(
    base_stream=base,
    evolution_pattern="eds",
    n_segments=3,         # Number of feature partitions
    overlap_ratio=0.5,    # Overlap length relative to stable period
    total_instances=10000
)
```

**Reference:** Hou, B. J., Zhang, L., & Zhou, Z. H. (2017). Learning with Feature Evolvable Streams. *NeurIPS*.

#### Consuming OpenFeatureStream Instances

Each instance from `OpenFeatureStream` carries a `feature_indices` attribute containing the global IDs of the currently active features:

```python
stream = OpenFeatureStream(
    base_stream=Electricity(),
    evolution_pattern="pyramid",
    d_min=2,
    d_max=8,
    total_instances=100
)

instance = stream.next_instance()
print(f"Feature values: {instance.x}")            # e.g., [0.056, 0.439]  (only active features)
print(f"Global indices: {instance.feature_indices}") # e.g., [0, 1]        (which features these are)
print(f"Active dimension: {len(instance.x)}")       # Changes over time
```

### 6.3 TrapezoidalStream — NaN-Based Trapezoidal Evolution

`TrapezoidalStream` maintains a **fixed vector size** equal to `d_max`. Inactive features are filled with `np.nan`. This is useful for algorithms like OVFM that handle missing values natively without needing index metadata.

**Supported Modes:**

| Mode | Behavior |
|---|---|
| `random` | Features appear in random order, linear growth `d_min → d_max` |
| `ordered` | Features appear sequentially (0, 1, 2, ...), linear growth |
| `pyramid` | Features appear sequentially, then disappear (`d_min → d_max → d_min`) |

```python
from openmoa.stream import TrapezoidalStream
from openmoa.datasets import Spambase

base = Spambase()
stream = TrapezoidalStream(
    base_stream=base,
    d_min=5,
    d_max=57,
    evolution_mode="random",
    total_instances=4601,
    random_seed=42
)

instance = stream.next_instance()
print(f"Vector size: {len(instance.x)}")  # Always 57
print(f"Active features: {np.count_nonzero(~np.isnan(instance.x))}")
print(f"Missing (NaN) features: {np.count_nonzero(np.isnan(instance.x))}")
```

### 6.4 CapriciousStream — NaN-Based Stochastic Missing Features

`CapriciousStream` simulates the Capricious Data Stream (CDS) paradigm with a fixed-dimension NaN representation. At each time step, each feature has a `missing_ratio` probability of being replaced with `NaN`.

```python
from openmoa.stream import CapriciousStream
from openmoa.datasets import German

base = German()
stream = CapriciousStream(
    base_stream=base,
    missing_ratio=0.5,    # 50% of features missing per instance
    min_features=1,       # At least 1 feature always present
    total_instances=1000,
    random_seed=42
)

instance = stream.next_instance()
# instance.x has fixed length d_max, with random NaN entries
```

### 6.5 EvolvableStream — NaN-Based Multi-Phase Evolution

`EvolvableStream` implements the Evolvable Data Stream (EDS) paradigm with fixed-dimension NaN representation. The feature space evolves through `n` segments with configurable overlap periods.

```python
from openmoa.stream import EvolvableStream
from openmoa.datasets import Adult

base = Adult()
stream = EvolvableStream(
    base_stream=base,
    n_segments=3,
    overlap_ratio=0.5,
    total_instances=10000,
    random_seed=42
)

instance = stream.next_instance()
# instance.x has fixed size, with NaN for features not in the current segment
```

### 6.6 Choosing the Right Wrapper

| Your Algorithm Handles... | Use This Wrapper | Representation |
|---|---|---|
| Variable-length sparse input with index metadata | `OpenFeatureStream` | Subsetted vector + `feature_indices` |
| Fixed-dimension input with NaN as missing indicator | `TrapezoidalStream` (for TDS) | Fixed vector with NaN |
| Fixed-dimension input with NaN as missing indicator | `CapriciousStream` (for CDS) | Fixed vector with NaN |
| Fixed-dimension input with NaN as missing indicator | `EvolvableStream` (for EDS) | Fixed vector with NaN |

### 6.7 Combining Wrappers

You can chain wrappers together. A common pattern is to shuffle a static dataset first, then apply feature evolution:

```python
from openmoa.stream import ShuffledStream, OpenFeatureStream
from openmoa.datasets import Magic04

# Step 1: Load and shuffle the static dataset
base = Magic04()
shuffled = ShuffledStream(base_stream=base, random_seed=42)

# Step 2: Apply feature evolution
evolving_stream = OpenFeatureStream(
    base_stream=shuffled,
    evolution_pattern="pyramid",
    d_min=2,
    d_max=10,
    total_instances=shuffled.get_num_instances()
)
```

---

## 7. Complete Example: End-to-End Streaming Pipeline

Here is a complete example that demonstrates loading a dataset, applying feature evolution, training an online learner, and evaluating its performance:

```python
from openmoa.datasets import Electricity
from openmoa.stream import OpenFeatureStream
from openmoa.classifier import OASFClassifier
from openmoa.evaluation import ClassificationEvaluator

# 1. Create a base data stream
base_stream = Electricity()
schema = base_stream.get_schema()

# 2. Wrap with feature evolution (incremental features)
evolving_stream = OpenFeatureStream(
    base_stream=base_stream,
    evolution_pattern="incremental",
    d_min=2,
    total_instances=10000
)

# 3. Create a learner designed for evolving features
learner = OASFClassifier(schema=schema)
evaluator = ClassificationEvaluator(schema=schema)

# 4. Test-then-train loop
while evolving_stream.has_more_instances():
    instance = evolving_stream.next_instance()
    prediction = learner.predict(instance)
    learner.train(instance)
    evaluator.update(instance.y_index, prediction)

print(f"Accuracy under feature evolution: {evaluator.accuracy():.2f}%")
```

---

## 8. Summary: Stream Types at a Glance

| Stream Class | Source | Key Feature |
|---|---|---|
| `Electricity`, `Covtype`, ... | Built-in datasets | Auto-download, ready to use |
| `ARFFStream` | ARFF files | Native MOA format |
| `CSVStream` | CSV files | Flexible delimiter and type inference |
| `LibsvmStream` | LIBSVM files | Sparse format for high-dimensional data |
| `NumpyStream` | NumPy arrays | Direct from Python objects |
| `TorchClassifyStream` | PyTorch datasets | Deep learning integration |
| `RandomTreeGenerator`, `SEA`, ... | Synthetic generators | Infinite, configurable streams |
| `DriftStream` | Composite | Concept drift simulation |
| `ConcatStream` | Composite | Chain multiple streams |
| `ShuffledStream` | Wrapper | Randomize instance order |
| `OpenFeatureStream` | Wrapper | Feature evolution (sparse-aware) |
| `TrapezoidalStream` | Wrapper | Feature evolution (NaN, TDS) |
| `CapriciousStream` | Wrapper | Feature evolution (NaN, CDS) |
| `EvolvableStream` | Wrapper | Feature evolution (NaN, EDS) |

---

## Next Steps

Now that you understand how data streams work in OpenMOA, you can explore:

- **Classification Algorithms** — Learn about the 30+ classifiers available, including algorithms designed for evolving feature spaces (FESL, OASF, RSOL, OVFM)
- **Concept Drift Detection** — Discover how to detect and respond to distribution changes using ADWIN, DDM, EDDM, and more
- **Evaluation Methods** — Master prequential evaluation, windowed metrics, and multi-learner comparison
- **Advanced Topics** — Explore regression, anomaly detection, semi-supervised learning, and AutoML in streaming settings
