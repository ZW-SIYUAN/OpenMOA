# ---------------------------------------------------------------------------
# UOL classifiers – pure Python/NumPy, no Java or PyTorch at import time.
# ---------------------------------------------------------------------------
from ._fesl_classifier import FESLClassifier
from ._oasf_classifier import OASFClassifier
from ._rsol_classifier import RSOLClassifier
from ._orf3v_classifier import ORF3VClassifier
from ._fobos_classifier import FOBOSClassifier
from ._ftrl_classifier import FTRLClassifier
from ._ovfm_classifier import OVFMClassifier
from ._oslmf_classifier import OSLMFClassifier

# ---------------------------------------------------------------------------
# MOA-based classifiers – loaded lazily so that importing this module does
# not require Java when only UOL classifiers are used.
# ---------------------------------------------------------------------------
_MOA_CLASSIFIERS = {
    "AdaptiveRandomForestClassifier": "._adaptive_random_forest",
    "EFDT": "._efdt",
    "HoeffdingTree": "._hoeffding_tree",
    "NaiveBayes": "._naive_bayes",
    "OnlineBagging": "._online_bagging",
    "OnlineAdwinBagging": "._online_adwin_bagging",
    "LeveragingBagging": "._leveraging_bagging",
    "PassiveAggressiveClassifier": "._passive_aggressive_classifier",
    "SGDClassifier": "._sgd_classifier",
    "KNN": "._knn",
    "StreamingGradientBoostedTrees": "._sgbt",
    "OzaBoost": "._oza_boost",
    "MajorityClass": "._majority_class",
    "NoChange": "._no_change",
    "OnlineSmoothBoost": "._online_smooth_boost",
    "StreamingRandomPatches": "._srp",
    "HoeffdingAdaptiveTree": "._hoeffding_adaptive_tree",
    "SAMkNN": "._samknn",
    "DynamicWeightedMajority": "._dynamic_weighted_majority",
    "CSMOTE": "._csmote",
    "WeightedkNN": "._weightedknn",
    "ShrubsClassifier": "._shrubs_classifier",
    "Finetune": "._finetune",
    "PLASTIC": "._plastic",
    # UOL classifiers that require PyTorch – loaded lazily to keep the
    # pure-NumPy subset importable without a torch installation.
    "OLD3SClassifier": "._old3s_classifier",
    "OWSSClassifier": "._owss_classifier",
}


def __getattr__(name: str):
    if name in _MOA_CLASSIFIERS:
        import importlib
        module = importlib.import_module(_MOA_CLASSIFIERS[name], package=__name__)
        cls = getattr(module, name)
        globals()[name] = cls  # cache so subsequent access is direct
        return cls
    raise AttributeError(f"module 'openmoa.classifier' has no attribute {name!r}")


__all__ = [
    "AdaptiveRandomForestClassifier",
    "EFDT",
    "HoeffdingTree",
    "NaiveBayes",
    "OnlineBagging",
    "OnlineAdwinBagging",
    "LeveragingBagging",
    "KNN",
    "PassiveAggressiveClassifier",
    "SGDClassifier",
    "StreamingGradientBoostedTrees",
    "OzaBoost",
    "MajorityClass",
    "NoChange",
    "OnlineSmoothBoost",
    "StreamingRandomPatches",
    "HoeffdingAdaptiveTree",
    "SAMkNN",
    "DynamicWeightedMajority",
    "CSMOTE",
    "WeightedkNN",
    "ShrubsClassifier",
    "Finetune",
    "FESLClassifier",
    "OASFClassifier",
    "RSOLClassifier",
    "ORF3VClassifier",
    "FOBOSClassifier",
    "FTRLClassifier",
    "OVFMClassifier",
    "OSLMFClassifier",
    "OLD3SClassifier",
    "OWSSClassifier",
    "PLASTIC",
]
