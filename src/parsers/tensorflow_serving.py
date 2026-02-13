"""
TensorFlow Serving Parser

Parses inference logs from TensorFlow Serving / TF Serving.
"""

import re
import json
from datetime import datetime
from typing import Optional, Dict, Any

from .base import BaseParser, InferenceLogEntry


class TensorFlowServingParser(BaseParser):
    """Parser for TensorFlow Serving inference logs"""
    
    # Example log formats:
    # I1234 12:30:45.123456 1234 main.cc:123] Processing model 'my_model' batch 8 -> 15.3ms
    # {"model_spec":{"name":"my_model","version":"1"},"request_id":"abc123",...}
    
    PATTERNS = {
        'text': r'(?P<level>[IWEF])(?P<date>\d{4})\s+(?P<time>[\d:.]+)\s+\d+\s+[\w.]+:\d+\]\s*(?P<message>.*)',
        'grpc': r'model_spec.*name.*latency',
    }
    
    def __init__(self):
        super().__init__()
        self._text_pattern = self._compile_pattern('text', self.PATTERNS['text'])
        self._inference_pattern = self._compile_pattern(
            'inference',
            r'Processing\s+model\s+[\'"]?(?P<model>[^\'"\s]+)[\'"]?.*?batch\s*(?P<batch>\d+).*?(?P<latency>[\d.]+)\s*ms'
        )
    
    def detect(self, line: str) -> bool:
        """Detect if this is a TensorFlow Serving log"""
        indicators = [
            "tensorflow", "tf_serving", "grpc", "model_spec",
            "tensorflow_serving", "serving_model"
        ]
        line_lower = line.lower()
        return any(ind in line_lower for ind in indicators)
    
    def parse_line(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse TensorFlow Serving log line"""
        line = line.strip()
        if not line:
            return None
        
        # Try JSON format first
        if line.startswith('{'):
            return self._parse_json(line)
        
        # Try text format
        return self._parse_text(line)
    
    def _parse_json(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse JSON format log (gRPC/REST response logs)"""
        try:
            data = json.loads(line)
            
            # Handle TF Serving model_spec format
            model_spec = data.get('model_spec', {})
            model_name = model_spec.get('name', data.get('model', 'unknown'))
            model_version = str(model_spec.get('version', {}).get('value', '')) or data.get('version')
            
            timestamp = self._parse_timestamp(data.get('timestamp', ''))
            if not timestamp:
                timestamp = datetime.now()
            
            # Extract metrics from various fields
            latency = self._safe_float(
                data.get('latency_ms') or 
                data.get('duration_ms') or 
                data.get('processing_time_ms', 0)
            )
            
            status = data.get('status', {})
            if isinstance(status, dict):
                status_code = status.get('code', 0)
                error_msg = status.get('message')
            else:
                status_code = 200 if str(status).lower() in ['ok', 'success'] else 500
                error_msg = None if status_code == 200 else str(status)
            
            return InferenceLogEntry(
                timestamp=timestamp,
                request_id=data.get('request_id', ''),
                model_name=model_name,
                model_version=model_version,
                batch_size=self._safe_int(data.get('batch_size', 1)),
                latency_ms=latency,
                queue_time_ms=self._safe_float(data.get('queue_time_ms')) if 'queue_time_ms' in data else None,
                compute_time_ms=self._safe_float(data.get('inference_time_ms')) if 'inference_time_ms' in data else None,
                status_code=status_code,
                error_message=error_msg,
                input_size_bytes=self._safe_int(data.get('request_size')) if 'request_size' in data else None,
                output_size_bytes=self._safe_int(data.get('response_size')) if 'response_size' in data else None,
                metadata={
                    'signature_name': model_spec.get('signature_name', 'serving_default'),
                    'platform': data.get('platform', 'tensorflow'),
                }
            )
        except json.JSONDecodeError:
            return None
    
    def _parse_text(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse Google-style text format log"""
        # First, check if it's the TF internal log format
        text_match = self._text_pattern.match(line)
        if text_match:
            message = text_match.group('message')
            
            # Try to extract inference info from the message
            inf_match = self._inference_pattern.search(message)
            if inf_match:
                groups = inf_match.groupdict()
                
                # Parse the date/time from TF log format (MMDD HH:MM:SS.ffffff)
                date_str = text_match.group('date')
                time_str = text_match.group('time')
                year = datetime.now().year
                try:
                    timestamp = datetime.strptime(
                        f"{year}{date_str} {time_str}", 
                        "%Y%m%d %H:%M:%S.%f"
                    )
                except ValueError:
                    timestamp = datetime.now()
                
                level = text_match.group('level')
                status_code = 200 if level in ['I', 'W'] else 500
                
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
                    error_message=None,
                    input_size_bytes=None,
                    output_size_bytes=None,
                    metadata={'level': level, 'raw_message': message}
                )
        
        return None
