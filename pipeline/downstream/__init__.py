"""下游模型注册表。import 本包会自动注册所有内置模型。

模型分类:
  开源权重模型: cig_fault, cig_channel
  领域算法模型: seismic_domain_model, horizon_tracker, facies_classifier,
               well_log_analyzer, attribute_extractor, traditional_code
  轻量分割:      sam (Otsu/flood fill, seismic优化)
"""
from .base import (  # noqa: F401
    DownstreamModel, register, get, available_names, available_models_desc,
)
from .sam import Sam
from .traditional_code import TraditionalCode
from .horizon_tracker import HorizonTracker
from .facies_classifier import FaciesClassifier
from .well_log_analyzer import WellLogAnalyzer
from .attribute_extractor import AttributeExtractor
from .cig_models import CigFaultDetector, CigChannelDetector


def bootstrap_defaults():
    """一次性注册所有内置下游模型。同名覆盖允许外部替换。"""
    # 轻量分割 (seismic优化)
    register(Sam())
    # 领域算法 (真实实现, 已验证)
    register(TraditionalCode())
    register(HorizonTracker())
    register(FaciesClassifier())
    register(WellLogAnalyzer())
    register(AttributeExtractor())
    # Only register models that perform real forward inference. The unfinished
    # SFM metadata adapter and runtime-trained synthetic RF remain experimental.
    try:
        register(CigFaultDetector())
    except Exception:
        pass
    try:
        register(CigChannelDetector())
    except Exception:
        pass
    # 领域预测模型：大 checkpoint，惰性加载
    try:
        from .seismic_domain_model import SeismicDomainDetector
        register(SeismicDomainDetector())
    except ImportError:
        pass


bootstrap_defaults()
