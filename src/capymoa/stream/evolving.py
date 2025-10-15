"""capymoa/stream/evolving.py - 特征演化流包装器"""
import numpy as np
from typing import Literal, Optional
from capymoa.stream import Stream
from capymoa.stream._stream import Schema
from capymoa.instance import LabeledInstance, RegressionInstance


class EvolvingFeatureStream(Stream):
    """将固定特征的数据流包装为特征演化流。
    
    支持多种演化模式：
    - pyramid: 特征数先增后减（OASF 论文）
    - incremental: 特征数单调增长
    - decremental: 特征数单调减少
    - tds: 梯形数据流，特征有"出生时间"（ORF3V 论文）
    - vfs: 完全随机缺失（ORF3V 论文）
    
    Example:
    >>> from capymoa.datasets import Electricity
    >>> from capymoa.stream import EvolvingFeatureStream
    >>> 
    >>> # Pyramid 模式（OASF）
    >>> stream = EvolvingFeatureStream(
    ...     base_stream=Electricity(),
    ...     evolution_pattern="pyramid",
    ...     d_min=2, d_max=6
    ... )
    >>> 
    >>> # VFS 模式（ORF3V）
    >>> stream = EvolvingFeatureStream(
    ...     base_stream=Electricity(),
    ...     evolution_pattern="vfs",
    ...     missing_ratio=0.75
    ... )
    >>> 
    >>> # TDS 模式（ORF3V）
    >>> stream = EvolvingFeatureStream(
    ...     base_stream=Electricity(),
    ...     evolution_pattern="tds",
    ...     d_min=1, d_max=6
    ... )
    """

    def __init__(
        self,
        base_stream: Stream,
        d_min: int = 2,
        d_max: Optional[int] = None,
        evolution_pattern: Literal["pyramid", "incremental", "decremental", "tds", "vfs"] = "pyramid",
        total_instances: int = 10000,
        feature_selection: Literal["prefix", "suffix", "random"] = "prefix",
        missing_ratio: float = 0.0,
        random_seed: int = 42
    ):
        """初始化特征演化流
        
        :param base_stream: 原始数据流（特征固定）
        :param d_min: 最小特征维度
        :param d_max: 最大特征维度（None 则使用原始特征数）
        :param evolution_pattern: 演化模式
            - pyramid: 前半程增长，后半程减少
            - incremental: 单调增长
            - decremental: 单调减少
            - tds: 梯形数据流（特征有出生时间）
            - vfs: 完全随机缺失（每个特征独立缺失）
        :param total_instances: 总样本数
        :param feature_selection: 特征选择方式（仅用于 pyramid/incremental/decremental）
        :param missing_ratio: VFS 模式下的特征缺失率（0.0-1.0）
        :param random_seed: 随机种子
        """
        self.base_stream = base_stream
        self.d_min = d_min
        
        # 获取原始特征数
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )
        
        self.evolution_pattern = evolution_pattern
        self.total_instances = total_instances
        self.feature_selection = feature_selection
        self.missing_ratio = missing_ratio
        self.random_seed = random_seed
        
        # 设置随机种子
        self._rng = np.random.RandomState(random_seed)
        
        # 当前时间步
        self._current_t = 0
        
        # 预计算演化序列
        if evolution_pattern in ["pyramid", "incremental", "decremental"]:
            self._dimension_schedule = self._generate_dimension_schedule()
            self._feature_indices_cache = self._generate_feature_indices()
        elif evolution_pattern == "tds":
            self._feature_offsets = self._generate_tds_offsets()
        elif evolution_pattern == "vfs":
            # VFS 模式不需要预计算
            pass
        
        # Schema
        self._schema = base_stream.get_schema()

    def _generate_dimension_schedule(self) -> np.ndarray:
        """生成特征维度演化序列"""
        dims = np.zeros(self.total_instances, dtype=int)
        
        if self.evolution_pattern == "pyramid":
            half = self.total_instances // 2
            # 前半程：从 d_min 增长到 d_max
            dims[:half] = np.linspace(self.d_min, self.d_max, half).astype(int)
            # 后半程：从 d_max 减少到 d_min
            dims[half:] = np.linspace(
                self.d_max, self.d_min, self.total_instances - half
            ).astype(int)
        
        elif self.evolution_pattern == "incremental":
            # 单调增长
            dims = np.linspace(self.d_min, self.d_max, self.total_instances).astype(int)
        
        elif self.evolution_pattern == "decremental":
            # 单调减少
            dims = np.linspace(self.d_max, self.d_min, self.total_instances).astype(int)
        
        return dims

    def _generate_feature_indices(self) -> list:
        """生成每个时间步的特征索引（用于 pyramid/incremental/decremental）"""
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
                raise ValueError(f"Unknown feature_selection: {self.feature_selection}")
            
            indices_list.append(indices)
        
        return indices_list

    def _generate_tds_offsets(self) -> np.ndarray:
        """生成 TDS 模式下的特征 offset（出生时间）"""
        # 将特征随机分配到 10 个时间段
        offsets = np.zeros(self.d_max, dtype=int)
        indices = self._rng.permutation(self.d_max)
        
        for i in range(self.d_max):
            feature_idx = indices[i]
            time_slot = i % 10
            offsets[feature_idx] = time_slot * (self.total_instances // 10)
        
        return offsets

    def _get_tds_indices(self, t: int) -> np.ndarray:
        """获取 TDS 模式下当前时刻的可用特征"""
        # 只返回已经"出生"的特征
        available = np.where(self._feature_offsets <= t)[0]
        return available

    def _get_vfs_indices(self, t: int) -> np.ndarray:
        """获取 VFS 模式下当前时刻的可用特征（随机缺失）"""
        # 每个特征独立地以 (1 - missing_ratio) 概率存在
        rng_t = np.random.RandomState(self.random_seed + t)
        mask = rng_t.rand(self.d_max) > self.missing_ratio
        available = np.where(mask)[0]
        return available

    def next_instance(self):
        """获取下一个实例（特征已演化）"""
        if not self.has_more_instances():
            return None
        
        # 从基础流获取原始实例
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 根据演化模式获取特征索引
        if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
            active_indices = self._feature_indices_cache[self._current_t]
        elif self.evolution_pattern == "tds":
            active_indices = self._get_tds_indices(self._current_t)
        elif self.evolution_pattern == "vfs":
            active_indices = self._get_vfs_indices(self._current_t)
        else:
            raise ValueError(f"Unknown evolution_pattern: {self.evolution_pattern}")
        
        # 如果没有可用特征，至少保留一个（避免空特征向量）
        if len(active_indices) == 0:
            active_indices = np.array([0])
        
        # 提取子集特征
        x_full = np.array(base_instance.x)
        x_subset = x_full[active_indices]
        
        # 创建新实例
        if self._schema.is_classification():
            modified_instance = LabeledInstance.from_array(
                self._schema,
                x_subset,
                base_instance.y_index
            )
        else:
            modified_instance = RegressionInstance.from_array(
                self._schema,
                x_subset,
                base_instance.y_value
            )
        
        self._current_t += 1
        return modified_instance

    def has_more_instances(self) -> bool:
        """检查是否还有更多实例"""
        return (
            self._current_t < self.total_instances 
            and self.base_stream.has_more_instances()
        )

    def restart(self):
        """重启流"""
        self.base_stream.restart()
        self._current_t = 0

    def get_schema(self) -> Schema:
        """返回 schema"""
        return self._schema

    def get_moa_stream(self):
        """自定义流不支持 MOA 加速"""
        return None

    def get_current_dimension(self) -> int:
        """获取当前特征维度（用于调试）"""
        if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
            if self._current_t < self.total_instances:
                return self._dimension_schedule[self._current_t]
            return self.d_min
        elif self.evolution_pattern == "tds":
            return len(self._get_tds_indices(self._current_t))
        elif self.evolution_pattern == "vfs":
            return len(self._get_vfs_indices(self._current_t))

    def get_dimension_schedule(self) -> Optional[np.ndarray]:
        """获取完整的维度演化序列（用于可视化）
        
        注意：仅适用于 pyramid/incremental/decremental 模式
        TDS 和 VFS 模式返回 None
        """
        if self.evolution_pattern in ["pyramid", "incremental", "decremental"]:
            return self._dimension_schedule.copy()
        return None
    
    def __len__(self) -> int:
        """返回流的长度"""
        return min(self.total_instances, len(self.base_stream))
class TrapezoidalStream(Stream):
    """专为 OVFM 设计的梯形数据流（TDS）
    
    特点：
    - 特征维度单调递增（从 d_min 到 d_max）
    - 特征按前缀顺序出现（0, 1, 2, ...）
    - 返回固定维度实例，缺失特征用 NaN 填充
    - 保证与 OVFM 的 TrapezoidalExpectationMaximization2 兼容
    
    Example:
        >>> from capymoa.datasets import Electricity
        >>> from capymoa.stream import OVFMTrapezoidalStream
        >>> 
        >>> stream = OVFMTrapezoidalStream(
        ...     base_stream=Electricity(),
        ...     d_min=2,
        ...     d_max=8,
        ...     total_instances=3000
        ... )
        >>> 
        >>> # 第一个实例只有前 2 个特征
        >>> inst1 = stream.next_instance()
        >>> print(len(inst1.x))  # 8 (但只有前2个有值，其余是NaN)
    """
    
    def __init__(
        self,
        base_stream: Stream,
        d_min: int = 2,
        d_max: Optional[int] = None,
        total_instances: int = 10000,
        num_phases: int = 10,
        random_seed: int = 42
    ):
        """初始化 OVFM 梯形流
        
        :param base_stream: 原始数据流
        :param d_min: 起始特征数
        :param d_max: 最终特征数（None 则使用全部）
        :param total_instances: 总样本数
        :param num_phases: 特征出现的阶段数（默认10，即每10%实例新增一批特征）
        :param random_seed: 随机种子
        """
        self.base_stream = base_stream
        self.d_min = d_min
        
        original_d = base_stream.get_schema().get_num_attributes()
        self.d_max = d_max if d_max is not None else original_d
        
        if self.d_max > original_d:
            raise ValueError(
                f"d_max ({self.d_max}) cannot exceed original feature count ({original_d})"
            )
        
        self.total_instances = total_instances
        self.num_phases = num_phases
        self.random_seed = random_seed
        
        self._current_t = 0
        self._schema = base_stream.get_schema()
        
        # 预计算每个特征的出生时间（前缀顺序）
        self._feature_birth_times = self._compute_birth_times()
    
    def _compute_birth_times(self) -> np.ndarray:
        """计算每个特征的出生时间（前缀顺序）"""
        birth_times = np.zeros(self.d_max, dtype=int)
        
        # 特征按索引顺序出生
        for i in range(self.d_max):
            # 计算该特征属于第几个阶段
            phase = int((i * self.num_phases) / self.d_max)
            # 该阶段的起始时间
            birth_times[i] = phase * (self.total_instances // self.num_phases)
        
        return birth_times
    
    def _get_active_features(self, t: int) -> int:
        """获取时刻 t 的活跃特征数（前 k 个特征）"""
        # 返回已经"出生"的特征数量
        return np.sum(self._feature_birth_times <= t)
    
    def next_instance(self):
        """获取下一个实例（固定维度，缺失用 NaN）"""
        if not self.has_more_instances():
            return None
        
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 获取当前活跃的特征数
        num_active = self._get_active_features(self._current_t)
        num_active = max(self.d_min, num_active)  # 至少 d_min 个特征
        
        # 创建固定维度的特征向量（全部用 NaN 初始化）
        x_full = np.full(self.d_max, np.nan)
        
        # 只填充活跃的特征
        x_base = np.array(base_instance.x)
        x_full[:num_active] = x_base[:num_active]
        
        # 创建新实例（固定维度）
        if self._schema.is_classification():
            modified_instance = LabeledInstance.from_array(
                self._schema,
                x_full,
                base_instance.y_index
            )
        else:
            modified_instance = RegressionInstance.from_array(
                self._schema,
                x_full,
                base_instance.y_value
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
    
    def __len__(self) -> int:
        return min(self.total_instances, len(self.base_stream))


class CapriciousStream(Stream):
    """专为 OVFM 设计的任意变化流（VFS）
    
    特点：
    - 特征随机缺失（每个样本独立）
    - 返回固定维度实例，缺失特征用 NaN 填充
    - 保证与 OVFM 的 OnlineExpectationMaximization 兼容
    
    Example:
        >>> from capymoa.datasets import Electricity
        >>> from capymoa.stream import OVFMCapriciousStream
        >>> 
        >>> stream = OVFMCapriciousStream(
        ...     base_stream=Electricity(),
        ...     missing_ratio=0.5,
        ...     total_instances=3000
        ... )
        >>> 
        >>> # 每个实例都是固定8维，但约50%的特征是NaN
        >>> inst = stream.next_instance()
        >>> print(len(inst.x))  # 8
        >>> print(np.sum(np.isnan(inst.x)))  # 约 4
    """
    
    def __init__(
        self,
        base_stream: Stream,
        missing_ratio: float = 0.5,
        total_instances: int = 10000,
        min_features: int = 1,
        random_seed: int = 42
    ):
        """初始化 OVFM 任意变化流
        
        :param base_stream: 原始数据流
        :param missing_ratio: 特征缺失率（0.0-1.0）
        :param total_instances: 总样本数
        :param min_features: 每个实例至少保留的特征数
        :param random_seed: 随机种子
        """
        self.base_stream = base_stream
        self.missing_ratio = missing_ratio
        self.total_instances = total_instances
        self.min_features = min_features
        self.random_seed = random_seed
        
        self._current_t = 0
        self._schema = base_stream.get_schema()
        self._num_features = base_stream.get_schema().get_num_attributes()
        self._rng = np.random.RandomState(random_seed)
    
    def _get_feature_mask(self, t: int) -> np.ndarray:
        """获取时刻 t 的特征掩码（True=保留，False=缺失）"""
        # 使用时间步作为种子，保证可重复性
        rng_t = np.random.RandomState(self.random_seed + t)
        
        # 生成掩码：每个特征以 (1 - missing_ratio) 概率保留
        mask = rng_t.rand(self._num_features) > self.missing_ratio
        
        # 确保至少有 min_features 个特征
        if np.sum(mask) < self.min_features:
            # 随机选择 min_features 个特征保留
            indices = rng_t.choice(
                self._num_features, 
                self.min_features, 
                replace=False
            )
            mask = np.zeros(self._num_features, dtype=bool)
            mask[indices] = True
        
        return mask
    
    def next_instance(self):
        """获取下一个实例（固定维度，缺失用 NaN）"""
        if not self.has_more_instances():
            return None
        
        base_instance = self.base_stream.next_instance()
        if base_instance is None:
            return None
        
        # 获取特征掩码
        mask = self._get_feature_mask(self._current_t)
        
        # 创建固定维度的特征向量
        x_base = np.array(base_instance.x)
        x_masked = x_base.copy()
        
        # 将缺失的特征设为 NaN
        x_masked[~mask] = np.nan
        
        # 创建新实例
        if self._schema.is_classification():
            modified_instance = LabeledInstance.from_array(
                self._schema,
                x_masked,
                base_instance.y_index
            )
        else:
            modified_instance = RegressionInstance.from_array(
                self._schema,
                x_masked,
                base_instance.y_value
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
    
    def get_current_missing_ratio(self) -> float:
        """获取当前实例的实际缺失率（用于调试）"""
        if self._current_t == 0:
            return 0.0
        mask = self._get_feature_mask(self._current_t - 1)
        return 1.0 - np.sum(mask) / len(mask)
    
    def __len__(self) -> int:
        return min(self.total_instances, len(self.base_stream))