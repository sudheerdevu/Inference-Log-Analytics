"""
Custom Log Parser

Flexible parser for custom log formats with configurable patterns.
"""

import re
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from .base import BaseParser, InferenceLogEntry


class CustomParser(BaseParser):
    """
    Flexible parser for custom inference log formats.
    
    Supports:
    - JSON logs with arbitrary field names
    - Regex-based text log parsing
    - Field mapping configuration
    """
    
    DEFAULT_FIELD_MAP = {
        'timestamp': ['timestamp', 'time', 'ts', 'datetime', 'date'],
        'request_id': ['request_id', 'req_id', 'id', 'trace_id', 'correlation_id'],
        'model_name': ['model_name', 'model', 'model_id', 'name'],
        'model_version': ['model_version', 'version', 'ver'],
        'batch_size': ['batch_size', 'batch', 'bs'],
        'latency_ms': ['latency_ms', 'latency', 'duration_ms', 'duration', 'time_ms', 'response_time'],
        'queue_time_ms': ['queue_time_ms', 'queue_time', 'queue_latency', 'wait_time'],
        'compute_time_ms': ['compute_time_ms', 'compute_time', 'inference_time', 'exec_time'],
        'status_code': ['status_code', 'status', 'code', 'http_status'],
        'error_message': ['error_message', 'error', 'err', 'message', 'msg'],
        'input_size_bytes': ['input_size_bytes', 'input_size', 'request_size'],
        'output_size_bytes': ['output_size_bytes', 'output_size', 'response_size'],
    }
    
    def __init__(self, 
                 field_map: Optional[Dict[str, List[str]]] = None,
                 regex_pattern: Optional[str] = None,
                 regex_groups: Optional[Dict[str, str]] = None,
                 delimiter: str = None):
        """
        Initialize custom parser.
        
        Args:
            field_map: Mapping of standard field names to possible field names in logs
            regex_pattern: Custom regex pattern for text logs
            regex_groups: Mapping of regex group names to standard field names
            delimiter: Delimiter for simple delimited logs (e.g., CSV, TSV)
        """
        super().__init__()
        self.field_map = {**self.DEFAULT_FIELD_MAP, **(field_map or {})}
        self.regex_pattern = regex_pattern
        self.regex_groups = regex_groups or {}
        self.delimiter = delimiter
        
        if regex_pattern:
            self._custom_pattern = self._compile_pattern('custom', regex_pattern)
        else:
            self._custom_pattern = None
    
    def detect(self, line: str) -> bool:
        """
        Custom parser always returns True as fallback.
        Override in subclass for specific detection logic.
        """
        return True
    
    def parse_line(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse log line using configured format"""
        line = line.strip()
        if not line:
            return None
        
        # Try JSON format
        if line.startswith('{'):
            return self._parse_json(line)
        
        # Try custom regex if configured
        if self._custom_pattern:
            return self._parse_regex(line)
        
        # Try delimiter-based parsing
        if self.delimiter:
            return self._parse_delimited(line)
        
        # Try generic key=value parsing
        return self._parse_key_value(line)
    
    def _parse_json(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse JSON format with flexible field mapping"""
        try:
            data = json.loads(line)
            return self._build_entry(data)
        except json.JSONDecodeError:
            return None
    
    def _parse_regex(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse using custom regex pattern"""
        match = self._custom_pattern.search(line)
        if not match:
            return None
        
        groups = match.groupdict()
        
        # Map regex groups to standard fields
        data = {}
        for std_field, regex_group in self.regex_groups.items():
            if regex_group in groups:
                data[std_field] = groups[regex_group]
        
        # Also include any groups that match standard field names
        for group_name, value in groups.items():
            if group_name not in data:
                data[group_name] = value
        
        return self._build_entry(data)
    
    def _parse_delimited(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse delimiter-separated values"""
        parts = line.split(self.delimiter)
        
        # Assume common order: timestamp, model, batch, latency, status
        data = {}
        if len(parts) >= 1:
            data['timestamp'] = parts[0]
        if len(parts) >= 2:
            data['model_name'] = parts[1]
        if len(parts) >= 3:
            data['batch_size'] = parts[2]
        if len(parts) >= 4:
            data['latency_ms'] = parts[3]
        if len(parts) >= 5:
            data['status_code'] = parts[4]
        
        return self._build_entry(data)
    
    def _parse_key_value(self, line: str) -> Optional[InferenceLogEntry]:
        """Parse key=value format"""
        # Pattern to match key=value or key="value" pairs
        kv_pattern = r'(\w+)\s*[=:]\s*["\']?([^"\'\s,]+)["\']?'
        matches = re.findall(kv_pattern, line)
        
        if not matches:
            return None
        
        data = {k: v for k, v in matches}
        return self._build_entry(data)
    
    def _find_field(self, data: Dict[str, Any], field_name: str) -> Any:
        """Find field value using field map"""
        possible_names = self.field_map.get(field_name, [field_name])
        
        for name in possible_names:
            # Try exact match
            if name in data:
                return data[name]
            # Try case-insensitive match
            for key in data:
                if key.lower() == name.lower():
                    return data[key]
        
        return None
    
    def _build_entry(self, data: Dict[str, Any]) -> InferenceLogEntry:
        """Build InferenceLogEntry from parsed data"""
        
        # Extract timestamp
        ts_value = self._find_field(data, 'timestamp')
        timestamp = self._parse_timestamp(str(ts_value)) if ts_value else datetime.now()
        
        # Extract status code and determine if success
        status_value = self._find_field(data, 'status_code')
        if status_value is not None:
            if isinstance(status_value, str):
                if status_value.lower() in ['success', 'ok', 'completed', 'true']:
                    status_code = 200
                elif status_value.lower() in ['error', 'fail', 'failed', 'false']:
                    status_code = 500
                else:
                    status_code = self._safe_int(status_value, 200)
            else:
                status_code = self._safe_int(status_value, 200)
        else:
            status_code = 200
        
        # Build metadata from remaining fields
        standard_fields = set()
        for field_names in self.field_map.values():
            standard_fields.update(field_names)
        
        metadata = {k: v for k, v in data.items() 
                   if k.lower() not in {f.lower() for f in standard_fields}}
        
        return InferenceLogEntry(
            timestamp=timestamp,
            request_id=str(self._find_field(data, 'request_id') or ''),
            model_name=str(self._find_field(data, 'model_name') or 'unknown'),
            model_version=str(self._find_field(data, 'model_version') or '') or None,
            batch_size=self._safe_int(self._find_field(data, 'batch_size'), 1),
            latency_ms=self._safe_float(self._find_field(data, 'latency_ms'), 0),
            queue_time_ms=self._safe_float(self._find_field(data, 'queue_time_ms')) if self._find_field(data, 'queue_time_ms') else None,
            compute_time_ms=self._safe_float(self._find_field(data, 'compute_time_ms')) if self._find_field(data, 'compute_time_ms') else None,
            status_code=status_code,
            error_message=str(self._find_field(data, 'error_message')) if self._find_field(data, 'error_message') else None,
            input_size_bytes=self._safe_int(self._find_field(data, 'input_size_bytes')) if self._find_field(data, 'input_size_bytes') else None,
            output_size_bytes=self._safe_int(self._find_field(data, 'output_size_bytes')) if self._find_field(data, 'output_size_bytes') else None,
            metadata=metadata
        )
