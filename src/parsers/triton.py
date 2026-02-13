"""
NVIDIA Triton Inference Server Parser

Parses inference logs from Triton Inference Server.
"""

import re
import json
from datetime import datetime
from typing import Optional, Dict, Any

from .base import BaseParser, InferenceLogEntry


class TritonParser(BaseParser):
    """Parser for NVIDIA Triton Inference Server logs"""
    
    # Example log formats:
    # I0115 10:30:45.123456 1 http_server.cc:3344] HTTP request: model_name=resnet50 ...
    # {"model_name":"resnet50","model_version":"1","request_id":"abc","batch_size":8,...}
    
    PATTERNS = {
        'http': r'HTTP\s+request.*model_name=(?P<model>\w+).*',
        'grpc': r'GRPC\s+request.*model_name=(?P<model>\w+).*',
        'inference': r'Inference\s+request.*(?P<model>\w+).*latency.*(?P<latency>[\d.]+)',
    }
    
    def __init__(self):
        super().__init__()
        self._log_pattern = self._compile_pattern(
            'triton_log',
            r'(?P<level>[IWEF])(?P<date>\d{4})\s+(?P<time>[\d:.]+)\s+(?P<thread>\d+)\s+(?P<file>[\w.]+):(?P<line>\d+)\]\s*(?P<message>.*)'
        )
        self._inference_pattern = self._compile_pattern(
            'inference',
            r'model_name[=:]?\s*[\'"]?(?P<model>[^\s\'"]+)[\'"]?'
        )
        self._stats_pattern = self._compile_pattern(
            'stats',
            r'queue.*?(?P<queue>[\d.]+).*?compute.*?(?P<compute>[\d.]+)|latency.*?(?P<latency>[\d.]+)'
        )
    
    def detect(self, line: str) -> bool:
        """Detect if this is a Triton log"""
        indicators = [
            "triton", "inference_request", "model_repository",
            "http_server.cc", "grpc_server.cc", "tritonserver",
            "triton inference server", "model_control"
        ]
        line_lower = line.lower()
        return any(ind in line_lower for ind in indicators)
    
    def parse_line(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse Triton Inference Server log line"""
        line = line.strip()
        if not line:
            return None
        
        # Try JSON format first (Triton metrics output)
        if line.startswith('{'):
            return self._parse_json(line)
        
        # Try native Triton log format
        return self._parse_text(line)
    
    def _parse_json(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse JSON format log"""
        try:
            data = json.loads(line)
            
            timestamp = self._parse_timestamp(
                data.get('timestamp') or 
                data.get('time') or 
                ''
            )
            if not timestamp:
                timestamp = datetime.now()
            
            # Triton-specific fields
            model_name = data.get('model_name', data.get('model', 'unknown'))
            model_version = data.get('model_version', data.get('version', ''))
            
            # Handle Triton statistics format
            stats = data.get('inference_stats', {})
            success = stats.get('success', {})
            
            # Calculate latency from Triton stats if available
            queue_time = None
            compute_time = None
            total_latency = self._safe_float(data.get('latency_ms', 0))
            
            if 'queue' in stats:
                queue_time = self._safe_float(stats['queue'].get('total_time_ns', 0)) / 1e6
            if 'compute_infer' in stats:
                compute_time = self._safe_float(stats['compute_infer'].get('total_time_ns', 0)) / 1e6
            
            if total_latency == 0 and queue_time and compute_time:
                total_latency = queue_time + compute_time
            
            # Determine success/failure
            exec_count = self._safe_int(success.get('count', data.get('count', 1)))
            fail_count = self._safe_int(stats.get('fail', {}).get('count', 0))
            status_code = 200 if fail_count == 0 else 500
            
            return InferenceLogEntry(
                timestamp=timestamp,
                request_id=data.get('request_id', data.get('id', '')),
                model_name=model_name,
                model_version=str(model_version) if model_version else None,
                batch_size=self._safe_int(data.get('batch_size', 1)),
                latency_ms=total_latency,
                queue_time_ms=queue_time,
                compute_time_ms=compute_time,
                status_code=status_code,
                error_message=data.get('error') if status_code != 200 else None,
                input_size_bytes=self._safe_int(data.get('request_size')) if 'request_size' in data else None,
                output_size_bytes=self._safe_int(data.get('response_size')) if 'response_size' in data else None,
                metadata={
                    'backend': data.get('backend'),
                    'execution_count': exec_count,
                    'gpu_id': data.get('gpu_id'),
                    'priority': data.get('priority'),
                }
            )
        except json.JSONDecodeError:
            return None
    
    def _parse_text(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse Triton native log format"""
        log_match = self._log_pattern.match(line)
        if not log_match:
            return None
        
        groups = log_match.groupdict()
        message = groups.get('message', '')
        
        # Skip non-inference log messages
        if 'model' not in message.lower() and 'inference' not in message.lower():
            return None
        
        # Parse the date/time
        date_str = groups.get('date', '')
        time_str = groups.get('time', '')
        year = datetime.now().year
        try:
            timestamp = datetime.strptime(f"{year}{date_str} {time_str}", "%Y%m%d %H:%M:%S.%f")
        except ValueError:
            timestamp = datetime.now()
        
        # Extract model name
        model_match = self._inference_pattern.search(message)
        model_name = model_match.group('model') if model_match else 'unknown'
        
        # Extract stats if available
        latency = 0.0
        queue_time = None
        compute_time = None
        
        stats_match = self._stats_pattern.search(message)
        if stats_match:
            groups_stats = stats_match.groupdict()
            if groups_stats.get('latency'):
                latency = self._safe_float(groups_stats['latency'])
            if groups_stats.get('queue'):
                queue_time = self._safe_float(groups_stats['queue'])
            if groups_stats.get('compute'):
                compute_time = self._safe_float(groups_stats['compute'])
        
        # Extract other values using regex
        batch_match = re.search(r'batch[_\s]*(?:size)?[=:]\s*(\d+)', message, re.I)
        batch_size = int(batch_match.group(1)) if batch_match else 1
        
        version_match = re.search(r'version[=:]\s*[\'"]?(\d+)[\'"]?', message, re.I)
        version = version_match.group(1) if version_match else None
        
        level = groups.get('level', 'I')
        status_code = 200 if level in ['I', 'W'] else 500
        
        return InferenceLogEntry(
            timestamp=timestamp,
            request_id='',
            model_name=model_name,
            model_version=version,
            batch_size=batch_size,
            latency_ms=latency,
            queue_time_ms=queue_time,
            compute_time_ms=compute_time,
            status_code=status_code,
            error_message=message if level == 'E' else None,
            input_size_bytes=None,
            output_size_bytes=None,
            metadata={
                'level': level,
                'file': groups.get('file'),
                'line': groups.get('line'),
                'thread': groups.get('thread'),
            }
        )
