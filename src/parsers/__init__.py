"""
Log Parsers Package

Provides parsers for various inference serving frameworks.
"""

from .base import BaseParser
from .onnx_runtime import ONNXRuntimeParser
from .tensorflow_serving import TensorFlowServingParser
from .triton import TritonParser
from .custom import CustomParser

__all__ = [
    "BaseParser",
    "ONNXRuntimeParser", 
    "TensorFlowServingParser",
    "TritonParser",
    "CustomParser",
]

# Parser registry
PARSERS = {
    "onnx_runtime": ONNXRuntimeParser,
    "tensorflow_serving": TensorFlowServingParser,
    "triton": TritonParser,
    "custom": CustomParser,
}


def get_parser(name: str) -> BaseParser:
    """Get parser by name"""
    if name not in PARSERS:
        raise ValueError(f"Unknown parser: {name}. Available: {list(PARSERS.keys())}")
    return PARSERS[name]()


def auto_detect_parser(log_line: str) -> str:
    """Attempt to auto-detect the appropriate parser for a log line"""
    if "triton" in log_line.lower() or "inference_request" in log_line.lower():
        return "triton"
    elif "tensorflow" in log_line.lower() or "tf_serving" in log_line.lower():
        return "tensorflow_serving"
    elif "onnxruntime" in log_line.lower() or "ort" in log_line.lower():
        return "onnx_runtime"
    return "custom"
