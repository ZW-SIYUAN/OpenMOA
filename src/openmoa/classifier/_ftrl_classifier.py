from __future__ import annotations
import numpy as np

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance

class FTRLClassifier(Classifier):
    """
    High-Performance FTRL-Proximal Classifier.
    
    Supports:
    1. Binary Classification (Logistic Regression)
    2. Multi-class Classification (Softmax Regression)
    
    Key Features:
    - Fully Vectorized: Uses NumPy for all mathematical operations.
    - Sparse-Aware: Efficient O(k) updates for high-dimensional sparse data (RCV1).
    - Wrapper-Aware: Compatible with OpenFeatureStream (TDS/CDS/EDS).
    
    Reference:
        McMahan, H. B. (2011). Follow-the-Regularized-Leader and Mirror Descent:
        Equivalence Theorems and L1 Regularization.
    
    Parameters:
        alpha: Alpha parameter (learning rate), typical range [0.005, 10.0]
        beta: Beta parameter (smoothing), typical range [0.1, 1.0]
        l1: L1 regularization (sparsity), typical range [0.0, 10.0]
        l2: L2 regularization (smoothing), typical range [0.0, 10.0]
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 0.1,
        beta: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        
        # 1. Task Detection
        if schema.is_regression():
            raise ValueError("FTRLClassifier does not support regression. Use FTRLRegressor instead.")
            
        self.n_classes = schema.get_num_classes()
        if self.n_classes == 2:
            self.task_type = "binary"
            self.n_outputs = 1
        else:
            self.task_type = "multiclass"
            self.n_outputs = self.n_classes

        # 2. Initialize Parameters
        # Use dense arrays for speed (even for RCV1, 47k floats is negligible memory)
        # Shape: (n_features, n_outputs)
        self.n_features = schema.get_num_attributes()
        np.random.seed(random_seed)
        
        # z: accumulated gradients - sigma * w
        self.z = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)
        # n: accumulated sum of squared gradients
        self.n = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)
        # w: current weights
        self.w = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)

    def __str__(self):
        return f"FTRLClassifier(task={self.task_type}, alpha={self.alpha}, beta={self.beta}, l1={self.l1}, l2={self.l2})"

    def train(self, instance: Instance):
        """Vectorized FTRL-Proximal update."""
        # 1. Get Sparse Representation
        indices, values = self._get_sparse_x(instance)
        if len(indices) == 0: return

        # 2. Compute Prediction & Gradient
        # w_active shape: (n_active_features, n_outputs)
        w_active = self.w[indices] 
        
        # linear_pred shape: (n_outputs,)
        # distinct from dot, this is sum(w_i * x_i) for each output column
        linear_pred = np.dot(values, w_active) 
        
        grad = None
        
        if self.task_type == "binary":
            # Logistic Regression: p = sigmoid(w^T x)
            pred = 1.0 / (1.0 + np.exp(-np.clip(linear_pred[0], -50, 50)))
            y = instance.y_index
            diff = pred - y
            # Gradient w.r.t weights: (p - y) * x
            # Shape: (n_active, 1)
            grad_scalar = np.array([diff])
            grad = np.outer(values, grad_scalar)

        else: # Multiclass
            # Softmax Regression
            shift = linear_pred - np.max(linear_pred)
            exp_scores = np.exp(shift)
            probs = exp_scores / np.sum(exp_scores)
            
            y = instance.y_index
            # Gradient: p_j - y_j
            diff = probs.copy()
            diff[y] -= 1.0
            
            # Gradient matrix: (n_active, n_classes)
            grad = np.outer(values, diff)

        # 3. FTRL Core Update (Vectorized)
        # Update n: sum of squared gradients
        # Note: We update n ONLY for active features
        n_active = self.n[indices]
        g_sq = grad ** 2
        n_new = n_active + g_sq
        self.n[indices] = n_new
        
        # Compute Sigma: (sqrt(n_new) - sqrt(n_old)) / alpha
        sigma = (np.sqrt(n_new) - np.sqrt(n_active)) / self.alpha
        
        # Update z: z + g - sigma * w
        z_active = self.z[indices]
        z_new = z_active + grad - (sigma * w_active)
        self.z[indices] = z_new
        
        # 4. Proximal Step (Sparsity Induction)
        # w = 0 if |z| <= l1 else ...
        sign_z = np.sign(z_new)
        abs_z = np.abs(z_new)
        
        denom = (self.beta + np.sqrt(n_new)) / self.alpha + self.l2
        
        # Mask for non-zero weights (L1 Thresholding)
        active_mask = abs_z > self.l1
        
        new_w = np.zeros_like(w_active)
        # Only update weights that exceed the L1 threshold
        if np.any(active_mask):
            numerator = - (z_new[active_mask] - sign_z[active_mask] * self.l1)
            new_w[active_mask] = numerator / denom[active_mask]
            
        self.w[indices] = new_w

    def predict(self, instance: Instance):
        indices, values = self._get_sparse_x(instance)
        if len(indices) == 0:
            return 0
            
        w_active = self.w[indices]
        linear_pred = np.dot(values, w_active)
        
        if self.task_type == "binary":
            pred = 1.0 / (1.0 + np.exp(-np.clip(linear_pred[0], -50, 50)))
            return 1 if pred > 0.5 else 0
        else: # Multiclass
            return np.argmax(linear_pred)

    def predict_proba(self, instance: Instance) -> np.ndarray:
        indices, values = self._get_sparse_x(instance)
        
        w_active = self.w[indices] if len(indices) > 0 else np.zeros((1, self.n_outputs))
        
        # Handle case where indices might be empty -> dot product is 0
        if len(indices) > 0:
            linear_pred = np.dot(values, w_active)
        else:
            linear_pred = np.zeros(self.n_outputs)
            
        if self.task_type == "binary":
            p = 1.0 / (1.0 + np.exp(-np.clip(linear_pred[0], -50, 50)))
            return np.array([1 - p, p])
        else:
            shift = linear_pred - np.max(linear_pred)
            exp_scores = np.exp(shift)
            return exp_scores / np.sum(exp_scores)

    def _get_sparse_x(self, instance: Instance):
        """Unified sparse extractor (Wrapper-Aware)."""
        # Case 1: OpenFeatureStream (Wrapper)
        if hasattr(instance, "feature_indices"):
            return instance.feature_indices, instance.x

        # Case 2: Native Sparse
        if hasattr(instance, "x_index") and hasattr(instance, "x_value"):
            return instance.x_index, instance.x_value
        
        # Case 3: Dense (possibly with NaNs from Wrapper)
        x = instance.x
        if not isinstance(x, np.ndarray): x = np.array(x)
        
        # Filter 0 and NaN
        valid_mask = (x != 0) & (~np.isnan(x))
        indices = np.where(valid_mask)[0]
        values = x[indices]
        
        return indices, values

    def get_sparsity(self) -> float:
        """Percentage of zero weights."""
        if self.w.size == 0: return 1.0
        n_zeros = np.sum(np.abs(self.w) < 1e-10)
        return n_zeros / self.w.size