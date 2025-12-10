from __future__ import annotations
import numpy as np

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance

class RSOLClassifier(Classifier):
    """
    RSOL (Robust Sparse Online Learning) - Optimized Implementation.
    
    Optimizations:
    1. Vectorized L1,2 Norm Sparsity (No Python loops).
    2. Ring Buffer for Sliding Window (Avoids expensive np.roll).
    3. Auto-expanding Weights (Handles RCV1 dynamic features).
    4. Wrapper-Aware (Handles sparse input / global indices).
    
    Reference:
        Chen, Z., et al. (2024). Robust Sparse Online Learning... SDM.
    """

    def __init__(
        self,
        schema: Schema,
        lambda_param: float = 50.0, # Regularization (Sparsity)
        mu: float = 1.0,            # PA Smoothness
        L: int = 1000,              # Window Size
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        
        if schema.get_num_classes() != 2:
            raise ValueError("RSOL only supports Binary Classification.")

        self.lambda_param = lambda_param
        self.mu = mu
        self.L = L
        
        np.random.seed(random_seed)
        
        # Initialize W (Features x Window)
        # Start small, expand later
        initial_dim = schema.get_num_attributes()
        if initial_dim == 0: initial_dim = 100
        
        self.W = np.zeros((initial_dim, self.L))
        
        # Ring Buffer Pointer (Points to the 'newest' column)
        self.ptr = 0
        
        self.current_dim = 0
        self.t = 0

    def __str__(self):
        return f"RSOLClassifier(lambda={self.lambda_param}, mu={self.mu}, L={self.L})"

    def _ensure_dimension(self, target_dim):
        """Dynamically expand W if new features appear."""
        current_rows = self.W.shape[0]
        if target_dim > current_rows:
            new_rows = max(target_dim, int(current_rows * 1.5))
            new_W = np.zeros((new_rows, self.L))
            new_W[:current_rows, :] = self.W
            self.W = new_W

    def train(self, instance: Instance):
        self.t += 1
        
        # 1. Parse Input
        indices, values = self._get_sparse_x(instance)
        
        # Determine feature dimension
        if len(indices) > 0:
            d_current = int(np.max(indices) + 1)
        else:
            d_current = self.current_dim
            
        self._ensure_dimension(d_current)
        
        # Construct Dense Vector xt (relative to d_current)
        # Needed because RSOL operates on dense weight vectors w_s / w_padded
        xt = np.zeros(d_current)
        xt[indices] = values
        
        y = 1 if instance.y_index == 1 else -1
        
        # 2. Retrieve most recent weights
        # In Ring Buffer, the previous weight is at (ptr - 1)
        prev_ptr = (self.ptr - 1) % self.L
        w_prev_full = self.W[:, prev_ptr]
        
        # 3. PA Update (Theorems 3.1 & 3.2 Unified Logic)
        # Logic: w_new = w_prev + gamma * y * x
        # Whether incremental or decremental, we just need to align dimensions.
        # Since we use a global matrix W that grows, 'w_prev' is already padded or truncated conceptually.
        
        # Use valid weights up to d_current
        w_s = w_prev_full[:d_current]
        
        # Loss: max(0, 1 - y * w.x)
        margin = np.dot(w_s, xt)
        loss = max(0, 1 - y * margin)
        
        # Gamma
        norm_sq = np.linalg.norm(xt)**2
        denom = norm_sq + 1/(2*self.mu)
        gamma = loss / denom if denom > 0 else 0
        
        # Calculate new weight vector
        w_new = w_s + gamma * y * xt
        
        # 4. Store in Ring Buffer
        # Write to 'ptr' column
        self.W[:d_current, self.ptr] = w_new
        # Zero out any rows beyond d_current (if dimension shrunk) - Optional but clean
        if d_current < self.W.shape[0]:
            self.W[d_current:, self.ptr] = 0.0
            
        # 5. Apply L1,2 Sparsity (Vectorized Theorem 3.3)
        self._apply_l12_sparsity(d_current)
        
        # Advance pointer
        self.ptr = (self.ptr + 1) % self.L
        self.current_dim = d_current

    def predict(self, instance: Instance) -> int:
        prob = self.predict_proba(instance)[1]
        return 1 if prob > 0.5 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        indices, values = self._get_sparse_x(instance)
        
        if len(indices) == 0:
            return np.array([0.5, 0.5])
            
        # Get latest weights
        prev_ptr = (self.ptr - 1) % self.L
        w_pred = self.W[:, prev_ptr]
        
        # Sparse Dot Product
        # w_pred is dense (N,), values is (k,), indices is (k,)
        margin = 0.0
        # Safe indexing
        valid_mask = indices < w_pred.shape[0]
        if np.any(valid_mask):
            valid_idx = indices[valid_mask]
            valid_val = values[valid_mask]
            margin = np.dot(w_pred[valid_idx], valid_val)
            
        prob = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
        return np.array([1 - prob, prob])

    def _apply_l12_sparsity(self, active_rows):
        """
        Theorem 3.3: Row-wise L2 Norm Soft Thresholding.
        w_i = 0  if ||w_i|| <= lambda
        w_i = (1 - lambda/||w_i||) * w_i  otherwise
        """
        # We process ONLY the active rows to save time
        # Matrix slice: (active_rows, L)
        W_sub = self.W[:active_rows, :]
        
        # 1. Compute L2 Norm of each ROW
        # axis=1 means sum across columns (time steps)
        row_norms = np.linalg.norm(W_sub, axis=1)
        
        # 2. Identify rows to Zero
        zero_mask = row_norms <= self.lambda_param
        
        # 3. Identify rows to Shrink
        shrink_mask = ~zero_mask
        
        # Apply Zeroing
        self.W[:active_rows][zero_mask] = 0.0
        
        # Apply Shrinkage
        if np.any(shrink_mask):
            # Scaling factor: (1 - lambda / norm)
            scales = 1.0 - self.lambda_param / row_norms[shrink_mask]
            # Broadcast scale to all L columns
            # scale shape: (n_shrink, 1) * W_sub shape: (n_shrink, L)
            self.W[:active_rows][shrink_mask] *= scales[:, np.newaxis]

    def _get_sparse_x(self, instance: Instance):
        """Wrapper-compatible sparse extractor."""
        if hasattr(instance, "feature_indices"):
            return instance.feature_indices, instance.x
        if hasattr(instance, "x_index") and hasattr(instance, "x_value"):
            return instance.x_index, instance.x_value
        
        x = instance.x
        if not isinstance(x, np.ndarray): x = np.array(x)
        valid_mask = (x != 0) & (~np.isnan(x))
        indices = np.where(valid_mask)[0]
        values = x[indices]
        return indices, values
    
    def get_sparsity(self):
        # Sparsity of the latest weight vector
        prev_ptr = (self.ptr - 1) % self.L
        w_latest = self.W[:self.current_dim, prev_ptr]
        if w_latest.size == 0: return 1.0
        n_zeros = np.sum(np.abs(w_latest) < 1e-10)
        return n_zeros / w_latest.size