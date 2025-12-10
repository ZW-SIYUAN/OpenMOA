from __future__ import annotations
import numpy as np

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance

class OASFClassifier(Classifier):
    """
    OASF (Online Active Sparse Feature learning) - Optimized for Benchmark.
    
    Features:
    1. Handles Incremental/Decremental features via PA updates (Theorem 1 & 2).
    2. Enforces Group Sparsity via L1,2-norm on sliding window W.
    3. Auto-expands to handle dynamic feature indices from Wrapper.
    
    Reference:
        Chen, Z., et al. (2024). l1,2-Norm and CUR Decomposition... BigData.
    """

    def __init__(
        self,
        schema: Schema,
        lambda_param: float = 0.01, # Sparsity regularization
        mu: float = 1.0,            # PA smoothness (larger = more rigid)
        L: int = 100,               # Window size
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        
        if schema.get_num_classes() != 2:
            raise ValueError("OASF only supports Binary Classification.")

        self.lambda_param = lambda_param
        self.mu = mu
        self.L = L
        
        np.random.seed(random_seed)
        
        # W: Weight Matrix (Features x Window)
        # We start small and expand as needed to handle RCV1 efficiently
        initial_dim = schema.get_num_attributes()
        if initial_dim == 0: initial_dim = 100 # Safe default
        
        self.W = np.zeros((initial_dim, self.L))
        
        self.current_dim = 0
        self.t = 0

    def __str__(self):
        return f"OASFClassifier(lambda={self.lambda_param}, mu={self.mu}, L={self.L})"

    def _ensure_dimension(self, target_dim):
        """Dynamically expand W if new features appear."""
        current_rows = self.W.shape[0]
        if target_dim > current_rows:
            # Expand by doubling or exact fit, whichever is robust
            new_rows = max(target_dim, int(current_rows * 1.5))
            new_W = np.zeros((new_rows, self.L))
            new_W[:current_rows, :] = self.W
            self.W = new_W

    def train(self, instance: Instance):
        self.t += 1
        
        # 1. Parse Input (Handle Sparse/Wrapper)
        indices, values = self._get_sparse_x(instance)
        # Determine current feature dimension from input
        # In Wrapper context, max index + 1 is the conceptual dimension
        if len(indices) > 0:
            d_current = int(np.max(indices) + 1)
        else:
            d_current = self.current_dim # No features, keep old dim
            
        self._ensure_dimension(d_current)
        
        y = 1 if instance.y_index == 1 else -1
        
        # 2. Construct Dense Vector for calculation (OASF is dense-logic based)
        # Since we operate on the whole W column, we need a dense x representation
        # relative to d_current.
        xt = np.zeros(d_current)
        xt[indices] = values
        
        # 3. Update (Incremental vs Decremental)
        # Logic: Compare d_current vs self.current_dim
        # Case A: Decremental (Features disappeared) -> Theorem 1
        if self.current_dim >= d_current:
            # We use weights up to d_current
            w_s = self.W[:d_current, -1] # Last column, survival rows
            
            # Loss & Update
            loss = max(0, 1 - y * np.dot(w_s, xt))
            # Denom: ||x||^2 + 1/(2*mu)
            denom = np.linalg.norm(xt)**2 + 1/(2*self.mu)
            gamma = loss / denom if denom > 0 else 0
            
            w_new = w_s + gamma * y * xt
            
        # Case B: Incremental (Features appeared) -> Theorem 2
        else:
            # We pad old weights with zeros to match d_current
            w_padded = np.zeros(d_current)
            w_padded[:self.current_dim] = self.W[:self.current_dim, -1]
            
            # Loss & Update
            loss = max(0, 1 - y * np.dot(w_padded, xt))
            denom = np.linalg.norm(xt)**2 + 1/(2*self.mu)
            gamma = loss / denom if denom > 0 else 0
            
            # Update survival part and new part together
            w_new = w_padded + gamma * y * xt

        # 4. Slide Window & Store New Weight
        self.W = np.roll(self.W, -1, axis=1) # Shift left
        self.W[:, -1] = 0.0 # Clear last column
        self.W[:d_current, -1] = w_new # Set new weights
        
        # 5. Apply L1,2 Norm Sparsity (Group Lasso)
        # We shrink the entire ROW (feature history) if it's weak
        self._apply_l12_sparsity(d_current)
        
        # Update state
        self.current_dim = d_current

    def predict(self, instance: Instance) -> int:
        prob = self.predict_proba(instance)[1]
        return 1 if prob > 0.5 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        indices, values = self._get_sparse_x(instance)
        if len(indices) == 0:
            return np.array([0.5, 0.5])
            
        d_current = int(np.max(indices) + 1)
        # If model is smaller than data, we can only predict using known features
        # If model is larger, we use up to d_current
        valid_dim = min(d_current, self.W.shape[0])
        
        w_pred = self.W[:valid_dim, -1] # Use latest weights
        
        # Dot product
        # Only sum indices that exist in w_pred
        valid_mask = indices < valid_dim
        valid_indices = indices[valid_mask]
        valid_values = values[valid_mask]
        
        margin = 0.0
        # Manual sparse dot since w_pred is dense but input is sparse
        # Actually w_pred[valid_indices] works fine
        if len(valid_indices) > 0:
            margin = np.dot(w_pred[valid_indices], valid_values)
            
        prob = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
        return np.array([1 - prob, prob])

    def _apply_l12_sparsity(self, active_rows):
        """
        Theorem 3: L1,2 Mixed Norm Regularization.
        If ||w_row||_2 <= lambda, set row to 0.
        Else, shrink row.
        """
        # Optimized: Only iterate up to active_rows
        # Calculation: Compute norms of all rows (vectorized)
        
        # W_active: (active_rows, L)
        W_sub = self.W[:active_rows, :]
        
        # 1. Compute L2 norms of each row: Shape (active_rows,)
        row_norms = np.linalg.norm(W_sub, axis=1)
        
        # 2. Identify rows to zero out
        zero_mask = row_norms <= self.lambda_param
        
        # 3. Identify rows to shrink
        shrink_mask = ~zero_mask
        
        # Apply Zeroing
        self.W[:active_rows][zero_mask] = 0.0
        
        # Apply Shrinkage
        # Scale factor = 1 - lambda / norm
        if np.any(shrink_mask):
            scales = 1.0 - self.lambda_param / row_norms[shrink_mask]
            # Broadcast scale to columns
            self.W[:active_rows][shrink_mask] *= scales[:, np.newaxis]

    def _get_sparse_x(self, instance: Instance):
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
        """Sparsity of the most recent weight vector."""
        w_latest = self.W[:self.current_dim, -1]
        if w_latest.size == 0: return 1.0
        n_zeros = np.sum(np.abs(w_latest) < 1e-10)
        return n_zeros / w_latest.size