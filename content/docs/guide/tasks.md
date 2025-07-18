---
title: Tasks
weight: 2
---

OpenMOA provides reproducible baselines for the following open-environment streaming tasks where the feature space is non-stationary, growing, shrinking, or re-ordered over time.

### Task	

### Task-1 Streaming Anomaly Detection under Open Feature Spaces (SOAD)

Given:
- A stream data S = {Bt}t=1…T arriving in mini-batch form, where the dimension d_t of Bt ∈ ℝ^(n_t×d_t) grows open-endedly over time (allowing new features to suddenly appear and old features to silently disappear);
- Each sample x_t ∈ Bt is only partially observable upon arrival (open-world partial observability);
- User-specified "annotation budget" k (up to k labels queried per round).

Goal:
At any given time τ, output a real-time updated anomaly detector f_τ, ensuring it
(1) Maximize the Area Under the Receiver Operating Characteristic Curve (AUROC) under the condition of using only ≤ k active labels;
(2) When expanding/contracting in the feature space, the model can continue to update without restarting.

Constraints:
- Single-round inference latency ≤ 10 ms;
- The memory usage increases sub-linearly with the feature dimension (sparse representation).

Baseline implementation:
- OpenOLAFD (IJCAI’25) – A sparse active online learning framework, including
  - Proactive query strategy: Uncertainty + Diversity + Budget-Aware Buffer;
  - Feature expansion module: Elastic Sparse Projection + Online Dictionary Learning;
- RSOL-SF (SDM’23) – Sparse Online Learning Based on Truncated Gradient and Robust Loss;
- ORFF-VS (AAAI’22) – Online Random Feature Forest, supporting dynamic dimension mapping.

Evaluation indicators:
- AUROC@τ, AUPRC@τ (averaged with time decay)
- Label Efficiency = AUROC / (#queried labels)
- Runtime & Peak Memory vs. d_max

───────────────────────────────

### Task-2 Online CUR Row-Sparse Matrix Sketching under Varying Dimensions (OCUR-VS)

Given:
- Stream matrix sequence {M_t ∈ ℝ^(n×d_t)}, where the column dimension d_t varies over time;
- User-specified compression rank r and row sparsity s (at most s non-zero coefficients per row).

Goal:
Outputting the CUR decomposition (C_t, U_t, R_t) in real-time such that
∥M_t – C_t U_t R_t∥_F ≤ ε and the rows of R_t satisfy the ℓ1,∞-mixed norm sparsity constraint.

Constraints:
- The complexity of a single update is O(nnz(M_t) + r^2 d_t);
- Re-decomposing the historical matrix is not allowed.

Baseline implementation:
- ℓ1,∞-OCUR (SDM’24) – Online Block Coordinate Descent + Row Soft Thresholding;
- ORFF-CUR – using ORFF as a random column/row sampler;
- Greedy-CUR – a classic leverage sampling baseline.

Evaluation indicators:
Relative reconstruction error = ∥M_t – CUR∥_F / ∥M_t∥_F
- Achievement rate of row sparsity (≥ 95% of rows satisfying s-sparsity)
- Update Throughput (rows/ms)

───────────────────────────────

### Task-3 Utilitarian Online Regression from Open-World Soft Sensing (UORS)

Given:
- Multi-source soft sensor stream X_t = [x_t^(1), …, x_t^(M)], where any source may go offline or be newly added at any time;
- The target variable y_t arrives with a delay only after the user submits a "utility query";
- The global utility function U(ŷ, y) is specified online by the user (e.g., energy consumption-accuracy tradeoff).

Goal:
Maintain a regressor h_t to ensure cumulative utility
Maximize ∑_{τ=1}^t U(h_τ(x_τ), y_τ), and ensure the model is robust to heterogeneous sensor drifts.

Constraints:
- The sensor dimension can be extended to 10^4, but only m_t ≪ M active sources are observed in each round;
- Support utility function hot swapping without retraining.

Baseline implementation:
- UOL-SS (ICDM’24) – Online Mirror Descent + Dynamic Weight Sharing + Sparse Group Lasso;
- ORFF-Util – a weighted version of Random Feature Forests, supporting online utility re-weighting;
- Ridge-Restart – Classical Ridge Regression + Periodic Restart Baseline.

Evaluation indicators:
- Cumulative Utility Regret vs. offline oracle
- Sensor Robustness = AUROC (Fault Detection)
- Model Update Latency

───────────────────────────────

### Task-4 Online Deep Representation Learning with Evolving Feature Spaces (ODR-EFS)

Given:
- High-dimensional stream data X_t ∈ ℝ^(n_t×d_t), where d_t varies over time;
- An optional semi-supervised signal (with weak labels for some samples).

Goal:
Joint learning
(1) Dynamic feature encoder ϕ_t: ℝ^{d_t} → ℝ^k, where k is fixed;
(2) The online classification/regression head ψ_t ensures that the accuracy of downstream tasks remains unchanged while keeping the model capacity under control.

Constraints:
- The number of encoder parameters grows sublinearly with d_t;
- Support catastrophic forgetting inhibition.

Baseline implementation:
- DV-RNN (TKDE’23) – Deep Variational Recurrent Neural Network + Scalable Feature Gating;
- ORFF-Encoder – Single-layer ORFF as a lightweight encoder;
- MLP-Reinit – Reinitialize the fully connected layer every time the dimension is expanded.

Evaluation indicators:
- Downstream task Accuracy@t
- Forgetting Measure = Accuracy@t – max_{τ≤t} Accuracy@τ
- Feature Expansion Cost (FLOPs)

───────────────────────────────

### Task-5 Online Random Feature Forest for Varying Feature Streams (ORFF-VF)

Given:
- Streaming binary classification data (x_t, y_t) containing only partial feature observations;
- Tree nodes and random features can be dynamically added or deleted in each round.

Goal:
Maintain the error of the ORFF classifier f_t to be ≤ ε, and ensure the model size is ≤ B MB.
Constraints:
- Tree depth ≤ 8, minimum sample size for leaf nodes = 10;
- Single-sample prediction latency ≤ 1 ms.

Baseline implementation:
- ORFF-VS (AAAI’22) – Original algorithm + Adaptive feature pool;
- ORFF-Budget – a pruned version with budget constraints;
- Hoeffding-Tree – a classic stream decision tree baseline.

Evaluation indicators:
- Error Rate@t
- Model Size (MB)
- Feature Utilization Ratio = Number of active features / Total number of observed features

### Task	

Streaming Anomaly Detection in Open Feature Spaces	
Detect anomalies while feature set expands or contracts; maintain low false-positive rate under concept drift.	
• Sparse Active Online Learning (IJCAI'25, default)


Fast Online CUR Decomposition under Varying Features	
Reconstruct and track the latent low-rank structure of a data stream when column (feature) set keeps changing.	
• ℓ1,∞-Mixed-Norm CUR (SDM'24)

Utilitarian Online Learning from Soft Sensing Streams	
Make real-time decisions when the incoming features are unreliable (soft) and the feature space is open.	
• Utilitarian OLLA (ICDM'24)

Dynamic Feature Selection & Model Adaptation	
Decide on-the-fly which new features to keep and how to update the model without full retraining.	
• Online Group LASSO wrapper



### Planned Extensions
Multilayer streaming graphs with dynamic node attributes.
Higher-order joint tensor factorization under open modes.
Federated OpenMOA: collaborative learning across institutions without sharing feature lists.
