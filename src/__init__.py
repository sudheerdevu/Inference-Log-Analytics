"""
Inference Log Analytics

A comprehensive toolkit for parsing, analyzing, and visualizing inference logs
from various ML serving frameworks.
"""

from .analyzer import InferenceLogAnalyzer

# Parser exports
from .parsers import (
    BaseParser,
    InferenceLogEntry,
    ONNXRuntimeParser,
    TensorFlowServingParser,
    TritonParser,
    CustomParser,
    auto_detect_parser,
    get_parser,
)

# Analyzer exports
from .analyzers import (
    LatencyAnalyzer,
    ThroughputAnalyzer,
    ErrorAnalyzer,
    AnomalyDetector,
)

# Visualizer exports
from .visualizers import Dashboard

# Alert exports
from .alerts import AlertManager, AlertRule, Alert, AlertSeverity

__version__ = "1.0.0"
__all__ = [
    # Main class
    "InferenceLogAnalyzer",
    
    # Parsers
    "BaseParser",
    "InferenceLogEntry",
    "ONNXRuntimeParser",
    "TensorFlowServingParser",
    "TritonParser",
    "CustomParser",
    "auto_detect_parser",
    "get_parser",
    
    # Analyzers
    "LatencyAnalyzer",
    "ThroughputAnalyzer",
    "ErrorAnalyzer",
    "AnomalyDetector",
    
    # Visualizers
    "Dashboard",
    
    # Alerts
    "AlertManager",
    "AlertRule",
    "Alert",
    "AlertSeverity",
]
