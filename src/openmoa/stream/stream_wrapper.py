import numpy as np
from typing import Literal, Optional, List
from openmoa.stream import Stream
from openmoa.stream._stream import Schema
from openmoa.instance import LabeledInstance, RegressionInstance

class OpenFeatureStream(Stream):
    """
    Wraps a fixed-feature data stream into an evolving feature stream.
    
    This class simulates various feature evolution scenarios where the feature space 
    changes over time (Concept Drift in Feature Space). It supports dynamic 
    insertion, deletion, and stochastic missing of features.

    A core capability of this class is the generation of 'Sparse-aware' Instances. 
    It attaches a `feature_indices` attribute to every generated Instance, representing 
    the Global Feature IDs. This allows downstream algorithms to correctly align 
    features regardless of their physical position in the array, resolving the 
    Index Shift problem common in varying feature spaces.

    Supported Evolution Patterns:
    - 'pyramid': Dimensions increase then decrease linearly.
    - 'incremental': Dimensions monotonically increase.
    - 'decremental': Dimensions monotonically decrease.
    - 'tds': Trapezoidal Data Stream. Features have distinct "birth times" (ordered or random).
    - 'cds': Capricious Data Stream. Features appear/disappear stochastically (Bernoulli trial).
    - 'eds': Evolvable Data Stream. Feature space evolves in sequential segments with overlapping transition periods.
    """

    def __init__(
        self,
        base_stream: Stream,
        d_min: int = 2,
        d_max: Optional[int] = None,
        evolution_pattern: Literal["pyramid", "incremental", "decremental", "tds", "cds", "eds"] = "pyramid",
        total_instances: int = 10000,
        feature_selection: Literal["prefix", "suffix", "random"] = "prefix",
        missing_ratio: float = 0.0,
        random_seed: int = 42,
        # TDS specific parameters
        tds_mode: Literal["random", "ordered"] = "random",
        # EDS specific parameters
        n_segments: int = 2,
        overlap_ratio: float = 1.0,
    ):
        """
        Initialize the OpenFeatureStream.

        :param base_stream: The original source stream (with fixed features).
        :param d_min: Minimum number of features (for deterministic patterns).
        :param d_max: Maximum number of features. If None, uses the original stream's dimension.
        :param evolution_pattern: The strategy governing feature evolution.
        :param total_instances: Total number of instances to generate.
        :param feature_selection: Selection strategy for 'pyramid'/'incremental'/'decremental'.
        :param missing_ratio: Probability of a feature being missing in 'cds' pattern (0.0 to 1.0).
        :param random_seed: Seed for reproducibility.
        :param tds_mode: (TDS only) 'random' assigns random birth times; 'ordered' assigns birth times by feature index.
        :param n_segments: (EDS only) Number of distinct feature partitions.
        :param overlap_ratio: (EDS only) Ratio of overlap period length to stable period length.
        """
        self.base_stream = base_stream
        self.d_min = d_min
        
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        if self.d_max > original_d:
            raise ValueError(f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})")
        
        self.evolution_pattern = evolution_pattern
        self.total_instances = total_instances
        self.feature_selection = feature_selection
        self.missing_ratio = missing_ratio
        self.random_seed = random_seed
        
        # TDS parameters
        self.tds_mode = tds_mode

        # EDS parameters
        self.n_segments = n_segments
        self.overlap_ratio = overlap_ratio

        self._rng = np.random.RandomState(random_seed)
        self._current_t = 0
        self._schema = base_stream.get_schema()

        # Pre-compute schedules for deterministic patterns
        if evolution_pattern in ["pyramid", "incremental", "decremental"]:
            self._dimension_schedule = self._generate_dimension_schedule()
            self._feature_indices_cache = self._generate_feature_indices()
            
        elif evolution_pattern == "tds":
            self._feature_offsets = self._generate_tds_offsets()
            
        elif evolution_pattern == "eds":
            if self.n_segments < 2:
                raise ValueError("n_segments must be >= 2 for EDS pattern")
            self._eds_partitions = self._generate_eds_partitions()
            self._eds_boundaries = self._calculate_eds_boundaries()
            
        # 'cds' is stochastic and calculated on-the-fly.

    def _generate_dimension_schedule(self) -> np.ndarray:
        """Generates the schedule for feature counts over time."""
        dims = np.zeros(self.total_instances, dtype=int)
        if self.evolution_pattern == "pyramid":
            half = self.total_instances // 2
            dims[:half] = np.linspace(self.d_min, self.d_max, half)
            dims[half:] = np.linspace(self.d_max, self.d_min, self.total_instances - half)
        elif self.evolution_pattern == "incremental":
            dims = np.linspace(self.d_min, self.d_max, self.total_instances)
        elif self.evolution_pattern == "decremental":
            dims = np.linspace(self.d_max, self.d_min, self.total_instances)
        return dims.astype(int)

    def _generate_feature_indices(self) -> list:
        """Pre-calculates feature indices for deterministic patterns."""
        indices_list = []
        for t in range(self.total_instances):
            d_current = self._dimension_schedule[t]
            if self.feature_selection == "prefix":
                indices = np.arange(d_current)
            elif self.feature_selection == "suffix":
                indices = np.arange(self.d_max - d_current, self.d_max)
            elif self.feature_selection == "random":
                rng_t = np.random.RandomState(self.random_seed + t)
                indices = rng_t.choice(self.d_max, d_current, replace=False)
                indices.sort()
            else:
                indices = np.arange(d_current)
            indices_list.append(indices)
        return indices_list
    
    def _generate_tds_offsets(self) -> np.ndarray:
        """Assigns birth times to features for TDS pattern."""
        offsets = np.zeros(self.d_max, dtype=int)
        
        # Divide timeline into 10 distinct "birth stages"
        time_step = self.total_instances // 10
        
        if self.tds_mode == "random":
            # Current TDS logic: Randomly assign features to birth stages
            indices = self._rng.permutation(self.d_max)
            for i in range(self.d_max):
                # Distribute features evenly across 10 buckets
                offsets[indices[i]] = (i % 10) * time_step
                
        elif self.tds_mode == "ordered":
            # New Ordered logic: Feature 0 born first, Feature N born last.
            # Behaves like 'incremental prefix'.
            for i in range(self.d_max):
                # Map index 'i' to one of the 10 buckets sequentially
                # bucket 0: first 10% of features, bucket 1: next 10%, etc.
                bucket = (i * 10) // self.d_max
                offsets[i] = bucket * time_step
                
        return offsets

    def _generate_eds_partitions(self) -> List[np.ndarray]:
        """Partitions the feature space sequentially into n segments for EDS."""
        # EDS is now strictly sequential (User request: removed random option)
        all_indices = np.arange(self.d_max)
        partitions = np.array_split(all_indices, self.n_segments)
        return [np.sort(p) for p in partitions]

    def _calculate_eds_boundaries(self) -> List[int]:
        """Calculates time boundaries for EDS stages (stable and overlapping)."""
        n = self.n_segments
        r = self.overlap_ratio
        
        # L is the length of a stable period. Total = n*L + (n-1)*r*L
        L = self.total_instances / (n + r * (n - 1))
        
        boundaries = []
        current_pos = 0.0
        total_stages = 2 * n - 1
        
        for i in range(total_stages):
            if i % 2 == 0: # Even index: Stable stage
                current_pos += L
            else:          # Odd index: Overlapping stage
                current_pos += L * r
            boundaries.append(int(current_pos))
            
        boundaries[-1] = self.total_instances
        return boundaries

    def _get_active_indices(self, t: int) -> np.ndarray:
        """
        Determines the set of active global feature IDs for the current time step.
        This is the core logic engine for all evolution patterns.
        """
        # 1. Deterministic Patterns
        if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
            if t < len(self._feature_indices_cache):
                return self._feature_indices_cache[t]
            return np.arange(self.d_min)
            
        # 2. TDS (Trapezoidal)
        elif self.evolution_pattern == "tds":
            return np.where(self._feature_offsets <= t)[0]
            
        # 3. CDS (Capricious Data Stream)
        elif self.evolution_pattern == "cds":
            rng_t = np.random.RandomState(self.random_seed + t)
            # Bernoulli trial: (1 - missing_ratio) probability of existence
            mask = rng_t.rand(self.d_max) > self.missing_ratio
            indices = np.where(mask)[0]
            if len(indices) == 0: indices = np.array([0]) 
            return indices
            
        # 4. EDS (Evolvable/Doubly-Streaming)
        elif self.evolution_pattern == "eds":
            stage_idx = 0
            for i, boundary in enumerate(self._eds_boundaries):
                if t < boundary:
                    stage_idx = i
                    break
            else:
                stage_idx = len(self._eds_boundaries) - 1
            
            if stage_idx % 2 == 0:
                # Even stage: Return indices from a single partition
                partition_idx = stage_idx // 2
                return self._eds_partitions[partition_idx]
            else:
                # Odd stage: Return union of adjacent partitions (Overlap)
                prev = (stage_idx - 1) // 2
                indices = np.concatenate([
                    self._eds_partitions[prev], 
                    self._eds_partitions[prev + 1]
                ])
                indices.sort()
                return indices

        return np.arange(self.d_max)

    def next_instance(self):
        """Retrieves the next instance with evolved features."""
        if not self.has_more_instances():
            return None
        
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # Step 1: Identify active global feature IDs
        active_indices = self._get_active_indices(self._current_t)
        
        # Step 2: Slice the physical data array
        x_full = np.array(base_instance.x)
        valid_indices = active_indices[active_indices < len(x_full)]
        if len(valid_indices) == 0:
            valid_indices = np.array([0])
            
        x_subset = x_full[valid_indices]
        
        # Step 3: Construct the new Instance
        if self._schema.is_classification():
            new_instance = LabeledInstance.from_array(
                self._schema, x_subset, base_instance.y_index
            )
        else:
            new_instance = RegressionInstance.from_array(
                self._schema, x_subset, base_instance.y_value
            )
            
        # Step 4: Attach Global IDs for alignment
        # This metadata enables downstream algorithms to map values to features correctly.
        new_instance.feature_indices = valid_indices
        
        self._current_t += 1
        return new_instance

    def has_more_instances(self) -> bool:
        return self._current_t < self.total_instances and self.base_stream.has_more_instances()

    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
    
    def get_schema(self) -> Schema:
        """Returns the schema of the original base stream (global feature space)."""
        return self._schema
    
    def get_moa_stream(self):
        return None


class TrapezoidalStream(Stream):
    """
    A fixed-dimension stream wrapper that simulates missing features using NaN.
    
    Unlike 'OpenFeatureStream' which changes the physical vector size, this class 
    maintains a constant vector size equal to `d_max`. Features that are currently 
    "inactive" (not yet born or already dead) are represented by `np.nan`.

    This is particularly useful for algorithms like OVFM that handle missing values 
    natively within a fixed schema.

    Supported Evolution Modes:
    1. 'random': Features appear one by one in a RANDOM order until d_max is reached.
                 (Linear growth: d_min -> d_max)
    2. 'ordered': Features appear sequentially by index (0, 1, 2...) until d_max is reached.
                  (Linear growth: d_min -> d_max)
    3. 'pyramid': Features appear sequentially up to d_max, then disappear sequentially.
                  (Triangular trend: d_min -> d_max -> d_min)
    """
    
    def __init__(
        self,
        base_stream: Stream,
        d_min: int = 2,
        d_max: Optional[int] = None,
        evolution_mode: Literal["random", "ordered", "pyramid"] = "random",
        total_instances: int = 10000,
        random_seed: int = 42
    ):
        """
        Initialize the TrapezoidalStream.

        :param base_stream: The source stream providing data.
        :param d_min: The minimum number of active features (starting dimension).
        :param d_max: The maximum dimension. If None, uses base_stream's dimension.
        :param evolution_mode: 
            - 'random': Random birth order, monotonic growth (TDS Random).
            - 'ordered': Sequential birth order, monotonic growth (TDS Ordered).
            - 'pyramid': Sequential birth/death, grows then shrinks.
        :param total_instances: Total length of the stream for schedule calculation.
        :param random_seed: Seed for reproducibility.
        """
        self.base_stream = base_stream
        self.d_min = d_min
        
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )
        
        self.evolution_mode = evolution_mode
        self.total_instances = total_instances
        self.random_seed = random_seed
        
        self._current_t = 0
        self._rng = np.random.RandomState(random_seed)
        
        # We reuse the base schema because the physical dimension is fixed (d_max)
        # However, conceptually, we ensure it matches the d_max expectation.
        self._schema = base_stream.get_schema()
        
        # --- Pre-computation ---
        # 1. Determine the priority order of features (Ranking)
        self._feature_ranking = self._generate_feature_ranking()
        
        # 2. Determine how many features are active at each time step (Schedule)
        self._dimension_schedule = self._generate_dimension_schedule()
    
    def _generate_feature_ranking(self) -> np.ndarray:
        """
        Determines the 'Priority Rank' of each feature.
        Features with lower rank index are activated first.
        """
        all_indices = np.arange(self.d_max)
        
        if self.evolution_mode == "random":
            # Shuffle indices: e.g., [5, 0, 9, ...] means Feature 5 is born first.
            self._rng.shuffle(all_indices)
            return all_indices
        
        elif self.evolution_mode in ["ordered", "pyramid"]:
            # Sequential indices: [0, 1, 2, ...] means Feature 0 is born first.
            return all_indices
            
        else:
            raise ValueError(f"Unknown evolution_mode: {self.evolution_mode}")

    def _generate_dimension_schedule(self) -> np.ndarray:
        """
        Calculates the number of active features (k) for every time step t.
        """
        dims = np.zeros(self.total_instances, dtype=int)
        
        if self.evolution_mode in ["random", "ordered"]:
            # Linear Growth: d_min -> d_max
            # This simulates the classic TDS "birth" process.
            dims = np.linspace(self.d_min, self.d_max, self.total_instances)
            
        elif self.evolution_mode == "pyramid":
            # Pyramid Trend: d_min -> d_max -> d_min
            half = self.total_instances // 2
            # Growth phase
            dims[:half] = np.linspace(self.d_min, self.d_max, half)
            # Shrinkage phase
            dims[half:] = np.linspace(self.d_max, self.d_min, self.total_instances - half)
            
        return dims.astype(int)
    
    def next_instance(self):
        """
        Returns an instance of fixed size `d_max`. 
        Inactive features are replaced with np.nan.
        """
        if not self.has_more_instances():
            return None
        
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 1. Get current active count (k) from schedule
        # Use simple clamping to handle cases where stream exceeds total_instances
        t_idx = min(self._current_t, self.total_instances - 1)
        num_active = self._dimension_schedule[t_idx]
        
        # 2. Identify WHICH features are active based on ranking
        # We select the top 'num_active' features from our ranking list
        active_indices = self._feature_ranking[:num_active]
        
        # 3. Create a canvas filled with NaN
        x_full = np.full(self.d_max, np.nan)
        
        # 4. Fill in the data for active features
        # We slice the base stream carefully to ensure dimension alignment
        x_base = np.array(base_instance.x)[:self.d_max]
        x_full[active_indices] = x_base[active_indices]
        
        # 5. Construct OpenMOA Instance
        if self._schema.is_classification():
            modified_instance = LabeledInstance.from_array(
                self._schema, x_full, base_instance.y_index
            )
        else:
            modified_instance = RegressionInstance.from_array(
                self._schema, x_full, base_instance.y_value
            )
        
        self._current_t += 1
        return modified_instance
    
    def has_more_instances(self) -> bool:
        return (
            self._current_t < self.total_instances 
            and self.base_stream.has_more_instances()
        )
    
    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
    
    def get_schema(self) -> Schema:
        return self._schema
    
    def get_moa_stream(self):
        return None


class CapriciousStream(Stream):
    """
    Simulates a Capricious Data Stream (CDS) for algorithms handling missing values (e.g., OVFM).
    
    Mechanism:
    - Maintains a FIXED feature dimension (d_max).
    - Randomly selects features to be "missing" based on a Bernoulli trial.
    - Missing features are replaced with `np.nan`.
    """
    
    def __init__(
        self,
        base_stream: Stream,
        d_max: Optional[int] = None,
        missing_ratio: float = 0.5,
        total_instances: int = 10000,
        min_features: int = 1,
        random_seed: int = 42
    ):
        """
        :param base_stream: Source stream.
        :param d_max: The fixed global dimension.
        :param missing_ratio: Probability of a feature being missing (0.0 to 1.0).
        :param total_instances: Simulation length.
        :param min_features: Minimum number of observed features per instance.
        :param random_seed: Seed for reproducibility.
        """
        self.base_stream = base_stream
        
        # Determine d_max
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        self.missing_ratio = missing_ratio
        self.total_instances = total_instances
        self.min_features = min_features
        self.random_seed = random_seed
        
        self._current_t = 0
        self._schema = base_stream.get_schema()
        # Removed redundant self._num_features, we use self.d_max consistently.
    
    def _get_feature_mask(self, t: int) -> np.ndarray:
        """Generates a boolean mask (True = Observed, False = Missing)."""
        # Ensure reproducibility per instance using time-based seed
        rng_t = np.random.RandomState(self.random_seed + t)
        
        # Bernoulli trial: True if random val > missing_ratio (i.e., feature is kept)
        # [Optimization] Use self.d_max instead of self._num_features
        mask = rng_t.rand(self.d_max) > self.missing_ratio
        
        # Safety check: Ensure at least 'min_features' are observed
        if np.sum(mask) < self.min_features:
            # Force enable random features to meet minimum requirement
            indices = rng_t.choice(
                self.d_max, 
                self.min_features, 
                replace=False
            )
            mask = np.zeros(self.d_max, dtype=bool)
            mask[indices] = True
        
        return mask
    
    def next_instance(self):
        if not self.has_more_instances():
            return None
        
        base = self.base_stream.next_instance()
        if not base:
            return None
        
        # [Fix 1] Call the helper method instead of rewriting logic
        mask = self._get_feature_mask(self._current_t)
        
        # [Optimization] Ensure float type for NaN compatibility
        x_base = np.array(base.x, dtype=float)
        
        # [Optimization] Safe slicing if base stream is larger than d_max
        if len(x_base) > self.d_max:
            x_base = x_base[:self.d_max]
            
        x_masked = x_base.copy()
        x_masked[~mask] = np.nan # Fill missing with NaN
        
        # [Fix 2] Support both Classification and Regression
        if self._schema.is_classification():
            new_instance = LabeledInstance.from_array(
                self._schema, x_masked, base.y_index
            )
        else:
            new_instance = RegressionInstance.from_array(
                self._schema, x_masked, base.y_value
            )
        
        self._current_t += 1
        return new_instance
    
    def has_more_instances(self) -> bool:
        return (
            self._current_t < self.total_instances 
            and self.base_stream.has_more_instances()
        )
    
    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
    
    def get_schema(self) -> Schema:
        return self._schema
    
    def get_moa_stream(self):
        return None
    


class EvolvableStream(Stream):
    """
    Simulates an N-phase Evolvable Data Stream (EDS) using a fixed-dimension 
    representation with NaN for missing values.

    This class is designed for algorithms like OVFM that expect a fixed global 
    feature space (d_max) but can handle missing values. It simulates the 
    evolution of feature spaces in sequential segments with overlapping transition periods.

    Evolution Logic (2n - 1 Stages):
    - Stage 0 (Stable): Partition 0 is active.
    - Stage 1 (Overlap): Partition 0 AND Partition 1 are active.
    - Stage 2 (Stable): Partition 1 is active.
    - ... and so on.

    The features are partitioned sequentially based on their indices.
    """

    def __init__(
        self,
        base_stream: Stream,
        d_max: Optional[int] = None,
        n_segments: int = 2,
        overlap_ratio: float = 1.0,
        total_instances: int = 10000,
        random_seed: int = 42
    ):
        """
        Initialize the EvolvableTrapezoidalStream (Sequential Only).

        :param base_stream: The original source stream.
        :param d_max: The fixed global dimension. If None, uses base_stream's dimension.
        :param n_segments: Number of distinct feature partitions (must be >= 2).
        :param overlap_ratio: Ratio of overlap period length to stable period length.
        :param total_instances: Total length of the stream for boundary calculation.
        :param random_seed: Seed for reproducibility (unused for partitioning now, kept for interface consistency).
        """
        self.base_stream = base_stream
        
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        if self.d_max > original_d:
            raise ValueError(f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})")
            
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")

        self.n_segments = n_segments
        self.overlap_ratio = overlap_ratio
        self.total_instances = total_instances
        self.random_seed = random_seed

        self._current_t = 0
        self._schema = base_stream.get_schema()
        
        # Pre-compute the feature sets (partitions)
        self._partitions = self._generate_partitions()
        
        # Pre-compute the timeline boundaries for stages
        self._stage_boundaries = self._calculate_boundaries()

    def _generate_partitions(self) -> List[np.ndarray]:
        """
        Divides the d_max features into n segments sequentially.
        E.g., if d_max=10, n=2 -> [0,1,2,3,4], [5,6,7,8,9]
        """
        all_indices = np.arange(self.d_max)
        # Split features into n roughly equal chunks sequentially
        partitions = np.array_split(all_indices, self.n_segments)
        return [np.sort(p) for p in partitions]

    def _calculate_boundaries(self) -> List[int]:
        """Calculates the time boundaries for the 2n-1 stages."""
        n = self.n_segments
        r = self.overlap_ratio
        
        # Calculate length of a stable period (L)
        # Total = n*L + (n-1)*r*L
        # Derived from: L * (n + r*(n-1)) = Total
        if (n + r * (n - 1)) == 0:
             L = 0
        else:
             L = self.total_instances / (n + r * (n - 1))
        
        boundaries = []
        current_pos = 0.0
        total_stages = 2 * n - 1
        
        for i in range(total_stages):
            if i % 2 == 0:
                # Even index: Stable stage
                current_pos += L
            else:
                # Odd index: Overlapping stage
                current_pos += L * r
            boundaries.append(int(current_pos))
            
        # Ensure the last boundary covers the exact end
        boundaries[-1] = self.total_instances
        return boundaries

    def _get_active_mask(self, t: int) -> np.ndarray:
        """Determines which features are active (True) or NaN (False) at time t."""
        # 1. Find current stage index
        stage_idx = 0
        for i, boundary in enumerate(self._stage_boundaries):
            if t < boundary:
                stage_idx = i
                break
        else:
            stage_idx = len(self._stage_boundaries) - 1
            
        # 2. Determine active partitions based on stage
        active_indices = []
        
        if stage_idx % 2 == 0:
            # Even Stage: Stable (Single Partition)
            # Stage 0 -> Partition 0; Stage 2 -> Partition 1
            p_idx = stage_idx // 2
            if p_idx < len(self._partitions):
                active_indices = self._partitions[p_idx]
        else:
            # Odd Stage: Overlap (Partition K + Partition K+1)
            # Stage 1 -> Part 0 & 1; Stage 3 -> Part 1 & 2
            prev = (stage_idx - 1) // 2
            if prev + 1 < len(self._partitions):
                active_indices = np.concatenate([
                    self._partitions[prev], 
                    self._partitions[prev + 1]
                ])
            else:
                # Fallback for edge cases, though calculation should prevent this
                active_indices = self._partitions[prev]

        # 3. Create Boolean Mask
        mask = np.zeros(self.d_max, dtype=bool)
        if len(active_indices) > 0:
            mask[active_indices] = True
        return mask

    def next_instance(self):
        """Returns the next instance with inactive features set to NaN."""
        if not self.has_more_instances():
            return None
        
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 1. Get Active Mask
        mask = self._get_active_mask(self._current_t)
        
        # 2. Prepare Data Canvas (Full of NaNs)
        x_full = np.full(self.d_max, np.nan)
        
        # 3. Fill Active Features
        x_base = np.array(base_instance.x, dtype=float)
        
        # Safety: handle case where base stream is larger/smaller than d_max
        limit = min(len(x_base), self.d_max)
        x_base_truncated = x_base[:limit]
        
        # Only copy features that exist in both base stream and mask
        # (Usually limit == d_max, so this is just x_full[mask] = x_base[mask])
        mask_truncated = mask[:limit]
        x_full[:limit][mask_truncated] = x_base_truncated[mask_truncated]
        
        # 4. Create Instance
        if self._schema.is_classification():
            new_instance = LabeledInstance.from_array(
                self._schema, x_full, base_instance.y_index
            )
        else:
            new_instance = RegressionInstance.from_array(
                self._schema, x_full, base_instance.y_value
            )
            
        self._current_t += 1
        return new_instance

    def has_more_instances(self) -> bool:
        return (
            self._current_t < self.total_instances 
            and self.base_stream.has_more_instances()
        )

    def restart(self):
        self.base_stream.restart()
        self._current_t = 0
    
    def get_schema(self) -> Schema:
        return self._schema
    
    def get_moa_stream(self):
        return None
    

# === 追加到 src/stream_wrapper.py 末尾 ===

class ShuffledStream(Stream):
    """
    A wrapper that buffers the entire base stream into memory and shuffles the order.
    Essential for static datasets (e.g., Magic04, Spambase) that might be sorted by label.
    
    WARNING: This consumes memory proportional to the dataset size. 
    Safe for UCI datasets (MBs), but DO NOT use for massive streams (GBs).
    """
    def __init__(self, base_stream: Stream, random_seed: int = 42):
        self.base_stream = base_stream
        self._schema = base_stream.get_schema()
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)
        
        # 1. Eagerly load all instances into memory
        self._instances = []
        while base_stream.has_more_instances():
            try:
                inst = base_stream.next_instance()
                if inst is not None:
                    self._instances.append(inst)
            except Exception:
                break
                
        self.n_instances = len(self._instances)
        
        # 2. Create a shuffled index map
        self._indices = np.arange(self.n_instances)
        self._rng.shuffle(self._indices)
        
        self._ptr = 0

    def has_more_instances(self):
        return self._ptr < self.n_instances

    def next_instance(self):
        if not self.has_more_instances():
            return None
        
        # Retrieve instance based on shuffled index
        real_idx = self._indices[self._ptr]
        instance = self._instances[real_idx]
        self._ptr += 1
        
        return instance

    def restart(self):
        """Resets the pointer and re-shuffles (optional behavior)."""
        self._ptr = 0
        # Optional: Reshuffle on restart if desired, or keep same shuffle
        # self._rng.shuffle(self._indices) 

    def get_schema(self):
        return self._schema
    
    def get_num_instances(self):
        """Crucial for downstream wrappers to know total length."""
        return self.n_instances
    
    def get_moa_stream(self):
        return None