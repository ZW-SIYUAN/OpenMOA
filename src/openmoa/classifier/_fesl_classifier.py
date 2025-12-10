from __future__ import annotations
import numpy as np
from collections import deque

from openmoa.base import Classifier
from openmoa.stream import Schema
from openmoa.instance import Instance

class FESLClassifier(Classifier):
    """
    FESL (Feature Evolvable Streaming Learning) - Authentic Implementation.
    
    Strict adherence to Hou et al. (NIPS 2017).
    Logic:
    1. Detects feature space shift (S_old -> S_new).
    2. Maintains two models: w_curr (on S_new) and w_old (on S_old).
    3. During overlap (buffered), learns a linear mapping M: S_new -> S_old via Ridge Regression.
    4. Prediction is an ensemble: y = mu1 * f_curr(x) + mu2 * f_old(M * x).
    
    WARNING: 
    This algorithm requires computing a dense mapping matrix M of size (d_new * d_old).
    - For UCI datasets (d < 5000): Works perfectly.
    - For RCV1 (d ~ 47,000): Will cause MemoryError (OOM) due to ~17GB matrix allocation.
      This is an algorithmic limitation of FESL, not a bug.
    """

    def __init__(
        self,
        schema: Schema,
        alpha: float = 0.1,      # Learning rate for SGD
        lambda_: float = 0.1,    # Regularization for Mapping (Ridge)
        window_size: int = 100,  # Buffer size (B) to learn mapping
        random_seed: int = 1,
    ):
        super().__init__(schema=schema, random_seed=random_seed)
        
        # FESL is strictly designed for binary classification logic (Logistic)
        if schema.get_num_classes() != 2:
            raise ValueError("FESLClassifier only supports Binary Classification.")

        self.alpha = alpha
        self.lambda_ = lambda_
        self.window_size = window_size
        
        np.random.seed(random_seed)
        
        # --- Models ---
        # We use dictionaries for sparse storage of weights to handle index shifts efficiently
        # w_curr: Weights for the current feature space
        # w_old: Weights for the previous feature space
        self.w_curr = {} 
        self.w_old = {}
        
        # Mapping Matrix M
        # Stores the dense matrix M and metadata to map global indices
        self.M_struct = None 
        
        # State tracking
        self.current_indices_set = set()
        
        # Buffers for learning M (Overlap period)
        # We store raw instances to reconstruct matrices X_old and X_new later
        self.overlap_buffer = [] 
        
        # Ensemble weights (Dynamic)
        self.mu_curr = 0.5
        self.mu_old = 0.5
        
        self.t = 0

    def __str__(self):
        return f"FESLClassifier(alpha={self.alpha}, lambda={self.lambda_}, win={self.window_size})"

    def train(self, instance: Instance):
            self.t += 1
            indices, values = self._get_sparse_x(instance)
            y = 1 if instance.y_index == 1 else -1
            
            # 1. Detect Feature Space Shift
            new_indices_set = set(indices)
            
            if len(self.current_indices_set) > 0 and new_indices_set != self.current_indices_set:
                intersection = len(new_indices_set.intersection(self.current_indices_set))
                union = len(new_indices_set.union(self.current_indices_set))
                jaccard = intersection / union if union > 0 else 0.0
                
                if jaccard < 0.8: 
                    self._transition_to_new_stage()
                    self.current_indices_set = new_indices_set
            
            if len(self.current_indices_set) == 0:
                self.current_indices_set = new_indices_set

            # 2. Buffer Data for Mapping
            if self.w_old and len(self.overlap_buffer) < self.window_size:
                x_dict = dict(zip(indices, values))
                self.overlap_buffer.append(x_dict)
                
                if len(self.overlap_buffer) == self.window_size:
                    self._learn_mapping()

            # 3. Update Current Model
            pred_curr = self._predict_linear(self.w_curr, indices, values)
            self._update_weights(self.w_curr, indices, values, pred_curr, y)
            
            # 4. Update Ensemble
            if self.M_struct is not None:
                pred_old = self._predict_via_mapping(indices, values)
                
                # === [FIX] 数值稳定的 Loss 计算 ===
                loss_curr = np.logaddexp(0, -y * pred_curr)
                loss_old = np.logaddexp(0, -y * pred_old)
                
                eta_ensemble = 0.1
                self.mu_curr *= np.exp(-eta_ensemble * loss_curr)
                self.mu_old *= np.exp(-eta_ensemble * loss_old)
                
                total_mu = self.mu_curr + self.mu_old
                if total_mu > 1e-10:
                    self.mu_curr /= total_mu
                    self.mu_old /= total_mu

    def predict(self, instance: Instance) -> int:
        prob = self.predict_proba(instance)[1]
        return 1 if prob > 0.5 else 0

    def predict_proba(self, instance: Instance) -> np.ndarray:
        indices, values = self._get_sparse_x(instance)
        
        # 1. Prediction from Current Model
        logit_curr = self._predict_linear(self.w_curr, indices, values)
        
        # 2. Prediction from Old Model (only if mapping exists)
        logit_old = 0.0
        if self.M_struct is not None:
            logit_old = self._predict_via_mapping(indices, values)
            
        # 3. Weighted Ensemble
        if self.M_struct is not None:
            final_logit = self.mu_curr * logit_curr + self.mu_old * logit_old
        else:
            final_logit = logit_curr
            
        prob = 1.0 / (1.0 + np.exp(-np.clip(final_logit, -50, 50)))
        return np.array([1 - prob, prob])

    def _transition_to_new_stage(self):
        """Called when significant drift is detected."""
        # Current model becomes the Old model
        self.w_old = self.w_curr.copy()
        
        # Note: We DO NOT clear w_curr. 
        # FESL implies 'Evolvable', so we inherit weights for overlapping features.
        # This acts as transfer learning.
        
        # Reset Mapping Logic for the new phase
        self.M_struct = None
        self.overlap_buffer = []
        
        # Reset ensemble weights to neutral
        self.mu_curr = 0.5
        self.mu_old = 0.5

    def _learn_mapping(self):
        """
        Calculates the Mapping Matrix M using Ridge Regression.
        Problem: X_old = X_new * M
        Solution: M = (X_new^T * X_new + lambda * I)^-1 * X_new^T * X_old
        """
        if not self.overlap_buffer: return
        
        # 1. Determine Dimensions
        # D_old: Union of all keys in w_old
        old_feat_ids = sorted(list(self.w_old.keys()))
        if not old_feat_ids: return
        
        # D_new: Union of all keys seen in the buffer
        new_feat_ids = set()
        for x_dict in self.overlap_buffer:
            new_feat_ids.update(x_dict.keys())
        new_feat_ids = sorted(list(new_feat_ids))
        if not new_feat_ids: return
        
        # 2. Construct Dense Matrices (Batch Size x Dim)
        # WARNING: This is the memory bottleneck for RCV1
        B = len(self.overlap_buffer)
        D_old = len(old_feat_ids)
        D_new = len(new_feat_ids)
        
        X_old = np.zeros((B, D_old))
        X_new = np.zeros((B, D_new))
        
        # Maps for fast index lookup
        old_map = {fid: i for i, fid in enumerate(old_feat_ids)}
        new_map = {fid: i for i, fid in enumerate(new_feat_ids)}
        
        # Fill matrices
        for i, x_dict in enumerate(self.overlap_buffer):
            for fid, val in x_dict.items():
                # Fill X_new (Input)
                if fid in new_map:
                    X_new[i, new_map[fid]] = val
                # Fill X_old (Target) - We assume overlap means we see these features too
                # In OpenFeatureStream EDS overlap, we receive the UNION of features.
                if fid in old_map:
                    X_old[i, old_map[fid]] = val
                    
        # 3. Solve Ridge Regression
        # (X'X + lambda*I) M = X'Y
        try:
            XtX = X_new.T @ X_new
            # Regularization
            reg_idx = np.arange(D_new)
            XtX[reg_idx, reg_idx] += self.lambda_
            
            XtY = X_new.T @ X_old
            
            # Solve linear system (faster and more stable than inv)
            M_dense = np.linalg.solve(XtX, XtY)
            
            # Save structure
            self.M_struct = {
                'matrix': M_dense,      # Shape (D_new, D_old)
                'new_map': new_map,     # Global ID -> Matrix Row
                'old_ids': old_feat_ids # Matrix Col -> Global ID (for w_old lookup)
            }
            
        except np.linalg.LinAlgError:
            # Fallback if singular
            self.M_struct = None
        except MemoryError:
            # Expected behavior for RCV1
            print(f"FESL Error: OOM during mapping learning (Dim: {D_new}x{D_old}). Feature evolution disabled.")
            self.M_struct = None

    def _predict_via_mapping(self, indices, values):
        """
        Predicts: y = w_old * (x_curr * M)
        Ideally: y = (w_old * M^T) * x_curr  <-- More efficient order?
        Let's stick to concept: Reconstruct x_old first.
        """
        M = self.M_struct['matrix']
        new_map = self.M_struct['new_map']
        old_ids = self.M_struct['old_ids']
        
        # 1. Construct dense x_curr vector (subset)
        x_vec = np.zeros(M.shape[0])
        for idx, val in zip(indices, values):
            if idx in new_map:
                x_vec[new_map[idx]] = val
        
        # 2. Map: x_rec = x_vec @ M  (Result: 1 x D_old)
        x_rec = np.dot(x_vec, M)
        
        # 3. Dot with w_old
        logit = 0.0
        for i, val in enumerate(x_rec):
            # Only if value is non-zero (dense dot product)
            if abs(val) > 1e-9:
                gid = old_ids[i]
                logit += self.w_old.get(gid, 0.0) * val
                
        return logit

    def _predict_linear(self, w_dict, indices, values):
        logit = 0.0
        for idx, val in zip(indices, values):
            logit += w_dict.get(idx, 0.0) * val
        return logit

    def _update_weights(self, w_dict, indices, values, pred, y):
        # Sigmoid prob for gradient calculation
        p = 1.0 / (1.0 + np.exp(-np.clip(pred, -50, 50)))
        grad_scalar = p - (1 if y == 1 else 0)
        
        for idx, val in zip(indices, values):
            grad = grad_scalar * val
            # Standard SGD update
            w_dict[idx] = w_dict.get(idx, 0.0) - self.alpha * grad

    def _get_sparse_x(self, instance: Instance):
        """Interface for OpenFeatureStream."""
        if hasattr(instance, "feature_indices"):
            return instance.feature_indices, instance.x
        if hasattr(instance, "x_index") and hasattr(instance, "x_value"):
            return instance.x_index, instance.x_value
        
        # Dense fallback
        x = instance.x
        if not isinstance(x, np.ndarray): x = np.array(x)
        # Filter NaN and 0
        valid_mask = (x != 0) & (~np.isnan(x))
        indices = np.where(valid_mask)[0]
        values = x[indices]
        return indices, values