from __future__ import annotations
import numpy as np
from typing import Literal

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance

class FOBOSClassifier(Classifier):
    """Forward-Backward Splitting (FOBOS) for Online Classification.

    An optimized implementation supporting both Binary and Multi-class classification.
    
    Key Optimizations:
    - Sparse Gradient Updates: Efficiently handles high-dimensional sparse data (e.g., Text).
    - Vectorized Regularization: Fast NumPy operations for L1/L2 proximal operators.
    - Unified Logic: Automatically switches between Logistic (Binary) and Softmax (Multi-class).

    Reference:
        Duchi, J., & Singer, Y. (2009). 
        Efficient Online and Batch Learning Using Forward Backward Splitting. 
        Journal of Machine Learning Research.

    Parameters:
        schema: Stream schema
        alpha: Learning rate (initial η)
        lambda_: Regularization strength
        regularization: 'l1' (Lasso), 'l2' (Ridge), 'l1_l2' (Group Lasso), or None
        step_schedule: 'sqrt' (1/sqrt(t)) or 'linear' (1/t)
        random_seed: Seed for weight initialization
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 1.0,
        lambda_: float = 0.001,
        regularization: Literal["l1", "l2", "l1_l2", "none"] = "l1",
        step_schedule: Literal["sqrt", "linear"] = "sqrt",
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        
        self.alpha = alpha
        self.lambda_ = lambda_
        self.regularization = regularization
        self.step_schedule = step_schedule
        
        # Determine task type based on schema
        self.n_classes = schema.get_num_classes()
        if self.n_classes == 2:
            self.task_type = "binary"
            # For binary, we only need one set of weights (log-odds)
            self.n_outputs = 1
        else:
            self.task_type = "multiclass"
            self.n_outputs = self.n_classes

        # Initialize Weights: Shape (n_features, n_outputs)
        self.n_features = schema.get_num_attributes()
        np.random.seed(random_seed)
        
        # Scale initialization for stability
        scale = 1.0 / np.sqrt(self.n_features) if self.n_features > 0 else 0.01
        self.W = np.random.randn(self.n_features, self.n_outputs) * scale
        
        self.t = 0  # Time step

    def __str__(self):
        return (f"FOBOSClassifier(task={self.task_type}, alpha={self.alpha}, "
                f"lambda={self.lambda_}, reg={self.regularization})")

    def train(self, instance: Instance):
        self.t += 1
        x_indices, x_values = self._get_sparse_x(instance)
        eta = self._get_learning_rate()
        
        # --- Step 1: Forward Step (Gradient Descent) ---
        if self.task_type == "binary":
            # 1. Compute Score: w^T x
            # W is (d, 1), so we access column 0
            score = np.dot(self.W[x_indices, 0], x_values)
            
            # 2. Prediction (Sigmoid)
            pred = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
            y_true = instance.y_index # 0 or 1
            
            # 3. Gradient: (p - y) * x
            grad_scalar = (pred - y_true)
            
            # 4. Sparse Update
            # Only update rows corresponding to non-zero features
            self.W[x_indices, 0] -= eta * grad_scalar * x_values

        else: # Multiclass
            # 1. Compute Scores: W^T x
            # W[x_indices] is (n_active, n_classes)
            # x_values is (n_active,)
            scores = self.W[x_indices].T @ x_values # Shape: (n_classes,)
            
            # 2. Softmax
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)
            
            # 3. Gradient Error Vector
            y_true = instance.y_index
            error_vector = probs.copy()
            error_vector[y_true] -= 1.0 # (p - y)
            
            # 4. Sparse Rank-1 Update
            # Update matrix = outer(x, error)
            # This updates active rows for ALL classes
            update_matrix = np.outer(x_values, error_vector)
            self.W[x_indices] -= eta * update_matrix

        # --- Step 2: Backward Step (Proximal Operator) ---
        if self.lambda_ > 0:
            self._apply_proximal_operator(eta)

    def predict(self, instance: Instance) -> int:
        x_indices, x_values = self._get_sparse_x(instance)
        
        if self.task_type == "binary":
            score = np.dot(self.W[x_indices, 0], x_values)
            # Prob > 0.5 implies Score > 0
            return 1 if score > 0 else 0
        else:
            scores = self.W[x_indices].T @ x_values
            return int(np.argmax(scores))

    def predict_proba(self, instance: Instance) -> np.ndarray:
        x_indices, x_values = self._get_sparse_x(instance)
        
        if self.task_type == "binary":
            score = np.dot(self.W[x_indices, 0], x_values)
            prob = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
            return np.array([1.0 - prob, prob])
        else:
            scores = self.W[x_indices].T @ x_values
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            return exp_scores / np.sum(exp_scores)

    def _get_sparse_x(self, instance: Instance):
            """
            Extracts indices and values, universally handling:
            1. OpenFeatureStream (Varying Feature Space with 'feature_indices')
            2. Wrapper Streams with NaN padding (Trapezoidal, Capricious, Evolvable)
            3. Native Sparse Instances (RCV1, etc.)
            """
            # Case 1: OpenFeatureStream
            # The instance has a physically smaller x, but carries global IDs in feature_indices
            if hasattr(instance, "feature_indices"):
                # 直接返回全局索引和对应的特征值
                return instance.feature_indices, instance.x

            # Case 2: Native Sparse Instance (e.g., from ARFF/LibSVM loader)
            if hasattr(instance, "x_index") and hasattr(instance, "x_value"):
                return instance.x_index, instance.x_value
            
            # Case 3: Dense Input (Standard or Wrapper with NaNs)
            x = instance.x
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            
            # 关键修复：Trapezoidal/Capricious/Evolvable 使用 NaN 表示特征缺失/未激活
            # 我们必须同时过滤掉 0 (稀疏优化) 和 NaN (防止计算崩溃)
            # 逻辑：只保留 (值不为0) 且 (值不是NaN) 的索引
            valid_mask = (x != 0) & (~np.isnan(x))
            
            indices = np.where(valid_mask)[0]
            values = x[indices]
            
            return indices, values

    def _get_learning_rate(self):
        if self.step_schedule == "sqrt":
            return self.alpha / np.sqrt(self.t)
        elif self.step_schedule == "linear":
            return self.alpha / self.t
        return self.alpha

    def _apply_proximal_operator(self, eta):
        """Applies vectorized regularization updates."""
        threshold = eta * self.lambda_

        if self.regularization == "l1":
            # Soft Thresholding: sign(w) * max(0, |w| - threshold)
            self.W = np.sign(self.W) * np.maximum(0.0, np.abs(self.W) - threshold)
            
        elif self.regularization == "l2":
            # Simple multiplicative decay
            self.W *= (1.0 - threshold)
            
        elif self.regularization == "l1_l2":
            # Group Lasso (Row Sparsity) - Mainly for Multiclass
            # Calculate L2 norm of each feature row
            row_norms = np.linalg.norm(self.W, axis=1, keepdims=True)
            # Avoid division by zero
            safe_norms = row_norms.copy()
            safe_norms[safe_norms == 0] = 1.0
            
            # Shrinkage factor: max(0, 1 - threshold / ||w_row||)
            shrinkage = np.maximum(0.0, 1.0 - threshold / safe_norms)
            
            # Apply to all columns
            self.W *= shrinkage