#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath('./src'))

from openmoa.stream import ARFFStream
from openmoa.evaluation import prequential_evaluation
from openmoa.classifier import FESLClassifier

FILE_PATH = r"C:/reposity clone/OpenMOA/datasets/FESL/australian_perm5.arff"

stream = ARFFStream(FILE_PATH)

# 关键修改：加入epoch参数
epoch = 1  # 不重复
nSmp = 690
B = 10
T1 = int(nSmp / 2)     # T1 = 345
T2 = nSmp - T1         # T2 = 345
max_instances = nSmp

fesl = FESLClassifier(
    schema=stream.schema,
    s1_feature_indices=list(range(42)),
    s2_feature_indices=list(range(42, 71)),
    overlap_size=10,
    switch_point=345,            # 345
    ensemble_method='selection',
    learning_rate_scale=1.0,
    random_seed=None
)

results = prequential_evaluation(stream, fesl,
                                 max_instances=max_instances,  # 改这里
                                 window_size=1,
                                 progress_bar=True)
print(f"Accuracy: {results['cumulative'].accuracy():.3f}%")


# 应该是 71（42+29）