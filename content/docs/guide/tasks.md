---
title: Tasks
weight: 2
---

List of Benchmark Tasks

## Task-1 Efficient & Interpretable Online Learning in Open, High-Dimensional Feature Spaces (SOOFS & RSOL)
### (1) Real-World Scenario
Imagine a city-scale disaster-monitoring system that continuously ingests data from crowd-sensing smartphones, IoT sensors, and social-media feeds.
- Devices come and go: a new iPhone introduces a fresh sensor modality; an old Android goes offline, removing its features; a viral hashtag spikes and vanishes within minutes.
- The feature space is open (arbitrarily evolving) and often high-dimensional (millions of hashtags, sensor channels, or user contexts).
- The system must predict in real time (e.g., “Is this region in danger?”) while keeping the model small enough to fit in edge-device memory and transparent enough for emergency managers to audit.

### (2) Formal Task Definition
Input
- Streaming instances: ${(x_t, y_t) \mid t \in [T]}$
  - $x_t \in ℝ^{d_t}$ where $d_t$ can grow or shrink at each round.
  - $y_t \in {+1, –1}$ (binary label, e.g., “danger” vs “safe”).

Knowledge Constraints
- Only one pass over the data (online).
- Model size must stay sub-linear in the total number of ever-seen features.

Objective

At each round t, output a sparse weight vector $w_t \in ℝ^{d_t}$ minimizing

$$Regret_T = Σ_{t=1}^T ℓ_t(w_t)  +  λ‖W_t‖_{1,∞} or ‖W_t‖_{1,2} $$

where
- $ℓ_t$ is the hinge loss: $\max(0,\; 1 - y_t(\mathbf{w}_t^\top \mathbf{x}_t))$.
- $‖W_t‖_{1,∞} or ‖W_t‖_{1,2}$ enforces row-wise sparsity (entire features pruned).
- Expected Influence: maximize prediction accuracy while keeping non-zero rows ≤ budget k (user-defined memory limit).

### (3) Loss Function & Optimization Target
Minimize sub-linear regret over T rounds:

$$min_{w_t}  Σ_{t=1}^T ℓ_t(w_t)   s.t.   ‖W‖_{1,∞} ≤ k$$

- Closed-form PA update handles dynamic dimension alignment.
- Online CUR decomposes the sliding-window weight matrix to actively retain the most informative instances, ensuring both efficiency and interpretability (original features retained or dropped en bloc).

───────────────────────────────

## Task-2 Efficient & Interpretable Online Learning in Open, Unknown Categories appear at any time Feature Spaces (OWSS)
### (1) Problem Statement
In feature-evolving and class-open online environments (e.g., semiconductor wafer inspection lines), sensors may be added or removed over time, and previously unseen defect types can emerge at any moment. OWSS seeks an online learner that, at each round t, receives a sample

$$(x_t, y_t), x_t \in ℝ^{d_t}, y_t ∈ Y_k ∪ Y_u$$,

where $d_t$ can increase (new sensors) or decrease (obsolete sensors), and $Y_u$ denotes unknown classes. The learner must decide on the fly whether to output a class label or to abstain.

### (2) Objective
Minimize the cumulative loss over a horizon T:

$$\min_{h_1,\dots,h_T}\sum_{t=1}^{T}\Bigl[\ell\bigl(y_t,h_t(x_t)\bigr)\cdot\mathbb{I}\bigl(h_t\ \text{predicts}\bigr)+c(x_t)\cdot\mathbb{I}\bigl(h_t\ \text{abstains}\bigr)\Bigr]$$,

subject to
- Universal representation: $z_t=\varphi(x_t) \in R^k$ with fixed $$k$$, updated via bipartite graph–GCN.
- predict: $r(z_t)\geq 0 $
- abstain: $r(z_t) \lt 0$
- Online update: parameters $(Θ, {f_i}, r)$ are updated once per sample without revisiting prior data.

### (3) Baselines
- closed-world online: OCO, OSLMF
- offline outlier detection: ECOD, LUNAR
- graph offline: GraSSNet
- open-world with pre-training: ORCA

───────────────────────────────

## Task-3 Oline Learning in Open Feature Space (OLD3S & ORF3V)

### (1) Real-world Problem
In a smart city environmental monitoring system, sensor networks continuously collect air quality data (e.g., PM2.5, CO₂, temperature, humidity). Over time:
- New sensors (e.g., NO₂ sensors) are deployed, introducing new features;
- Old sensors fail or are decommissioned, causing old features to vanish. The goal is to train an online classifier that predicts air quality levels (e.g., good/moderate/poor/hazardous) in real time, while adapting to the evolving feature space (new features emerging, old features disappearing) without retraining from scratch.

### (2) Mathematical Formulation
- Data Stream $\{(x_t, y_t)\}_{t=1}^{T}$, where $x_t \in \mathbb{R}^{d_t}$ is the feature vector at time step $t$ (its dimension $d_t$ changes over time), and $y_t \in \{1,2,3,4\}$ is the air-quality label.

- Feature-Space Evolution:
  - $T_1$ Phase: Feature space $S_1 \subseteq \mathbb{R}^{d_1}$ (e.g.\ PM$_{2.5}$, temperature).
  - $T_b$ Phase (Overlap): Feature space expands to $S_1 \times S_2 \subseteq \mathbb{R}^{d_1+d_2}$ (new NO₂ sensors added).
  - $T_2$ Phase: Only the new feature space $S_2 \subseteq \mathbb{R}^{d_2}$ remains (old sensors retired).

- Objective: Learn a sequence of classifiers $f_t_t^T$ that minimizes the cumulative loss:
  $$\min_{\{f_t\}} \frac{1}{T}\sum_{t=1}^{T} \ell\bigl(y_t, f_t(x_t)\bigr)$$,
  where $\ell$ is the cross-entropy loss, and each $f_t$ must adapt to the evolving feature space.

### (3) Baselines
Method 1: OLD3S

Core Idea: Use Variational Autoencoders (VAEs) to learn a shared latent representation between old and new feature spaces, and dynamically adjust model depth.

- Latent Space Alignment (during $T_b$)

Map $S_1$ and $S_2$ to a shared latent space $Z\subseteq\mathbb{R}^{z}$ via VAEs, minimizing:
$\mathcal{L}_{\text{rec}} = \bigl\|x_{S_1}-\mathrm{Dec}_{2\to 1}\bigl(z_{S_2}\bigr)\bigr\|^{2}\+\\mathrm{KL}\Bigl(Q\!\bigl(z_{S_1}\!\mid x_{S_1}\bigr)\big\|\; Q\bigl(z_{S_2}\mid x_{S_2}\bigr)\Bigr)$

- Online Ensemble (during $T_2$)

Predict via a weighted combination of the old and new classifiers:

$\hat{y}_{t} = p\cdot f_{S_1}\bigl(\tilde{x}_{S_1}\bigr) + (1-p)\cdot f_{S_2}\bigl(x_{S_2}\bigr)$,

where $p$ decays as the new model $f_{S_2}$ improves (exponential experts).


- Adaptive Depth

Use Hedge Backpropagation (HBP) to dynamically select optimal model depth $l^*$ balancing convergence speed and expressiveness.

### Planned Extensions
Multilayer streaming graphs with dynamic node attributes.
Higher-order joint tensor factorization under open modes.
Federated OpenMOA: collaborative learning across institutions without sharing feature lists.
