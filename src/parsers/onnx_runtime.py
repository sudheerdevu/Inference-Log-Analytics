"""
ONNX Runtime Parser

Parses inference logs from ONNX Runtime inference server.
"""

import re
import json
from datetime import datetime
from typing import Optional, Dict, Any

from .base import BaseParser, InferenceLogEntry


class ONNXRuntimeParser(BaseParser):
    """Parser for ONNX Runtime inference logs"""
    
    # Example log formats:
    # [2024-01-15 10:30:45.123] INFO: Session 'model_v1' inference completed in 15.3ms (batch=8)
    # {"timestamp":"2024-01-15T10:30:45.123Z","model":"model_v1","latency_ms":15.3,"status":"success"}
    
    PATTERNS = {
        'text': r'\[(?P<timestamp>[^\]]+)\]\s*(?P<level>\w+):\s*Session\s*[\'"](?P<model>[^\'"]+)[\'"]\s*inference\s*(?P<status>\w+).*?(?P<latency>[\d.]+)\s*ms.*?batch=(?P<batch>\d+)',
        'json': r'^\s*{.*"model".*"latency".*}\s*$',
    }
    
    def __init__(self):
        super().__init__()
        self._text_pattern = self._compile_pattern('text', self.PATTERNS['text'])
    
    def detect(self, line: str) -> bool:
        """Detect if this is an ONNX Runtime log"""
        indicators = [
            "onnxruntime", "ort", "session", "inference completed",
            "onnx_runtime", "InferenceSession"
        ]
        line_lower = line.lower()
        return any(ind in line_lower for ind in indicators)
    
    def parse_line(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse ONNX Runtime log line"""
        line = line.strip()
        if not line:
            return None
        
        # Try JSON format first
        if line.startswith('{'):
            return self._parse_json(line)
        
        # Try text format
        return self._parse_text(line)
    
    def _parse_json(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse JSON format log"""
        try:
            data = json.loads(line)
            
            timestamp = self._parse_timestamp(data.get('timestamp', ''))
            if not timestamp:
                timestamp = datetime.now()
            
            return InferenceLogEntry(
                timestamp=timestamp,
                request_id=data.get('request_id', data.get('session_id', '')),
                model_name=data.get('model', data.get('model_name', 'unknown')),
                model_version=data.get('version', data.get('model_version')),
                batch_size=self._safe_int(data.get('batch_size', data.get('batch', 1))),
                latency_ms=self._safe_float(data.get('latency_ms', data.get('latency', 0))),
                queue_time_ms=self._safe_float(data.get('queue_time_ms')) if 'queue_time_ms' in data else None,
                compute_time_ms=self._safe_float(data.get('compute_time_ms')) if 'compute_time_ms' in data else None,
                status_code=200 if data.get('status', '').lower() in ['success', 'ok', 'completed'] else 500,
                error_message=data.get('error'),
                input_size_bytes=self._safe_int(data.get('input_size')) if 'input_size' in data else None,
                output_size_bytes=self._safe_int(data.get('output_size')) if 'output_size' in data else None,
                metadata={
                    'provider': data.get('execution_provider', 'CPU'),
                    'graph_optimization_level': data.get('optimization_level'),
                    'intra_op_threads': data.get('intra_op_threads'),
                }
            )
        except json.JSONDecodeError:
            return None
    
    def _parse_text(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse text format log"""
        match = self._text_pattern.search(line)
        if not match:
            return None
        
        groups = match.groupdict()
        timestamp = self._parse_timestamp(groups.get('timestamp', ''))
        if not timestamp:
            timestamp = datetime.now()
        
        status_str = groups.get('status', '').lower()
        status_code = 200 if status_str in ['completed', 'success', 'ok'] else 500
        
        return InferenceLogEntry(
            timestamp=timestamp,
            request_id='',
            model_name=groups.get('model', 'unknown'),
            model_version=None,
            batch_size=self._safe_int(groups.get('batch', 1)),
            latency_ms=self._safe_float(groups.get('latency', 0)),
            queue_time_ms=None,
            compute_time_ms=None,
            status_code=status_code,
            error_message=None if status_code == 200 else f"Status: {status_str}",
            input_size_bytes=None,
            output_size_bytes=None,
            metadata={'level': groups.get('level', 'INFO')}
        )
