"""
Base Parser Module

Provides the abstract base class for all log parsers.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Iterator


@dataclass
class InferenceLogEntry:
    """Represents a single parsed inference log entry"""
    timestamp: datetime
    request_id: str
    model_name: str
    model_version: Optional[str]
    batch_size: int
    latency_ms: float
    queue_time_ms: Optional[float]
    compute_time_ms: Optional[float]
    status_code: int
    error_message: Optional[str]
    input_size_bytes: Optional[int]
    output_size_bytes: Optional[int]
    metadata: Dict[str, Any]
    
    @property
    def is_success(self) -> bool:
        return self.status_code == 200 or self.status_code == 0
    
    @property
    def total_time_ms(self) -> float:
        return self.latency_ms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "batch_size": self.batch_size,
            "latency_ms": self.latency_ms,
            "queue_time_ms": self.queue_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "input_size_bytes": self.input_size_bytes,
            "output_size_bytes": self.output_size_bytes,
            "metadata": self.metadata,
        }


class BaseParser(ABC):
    """Abstract base class for inference log parsers"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self._compiled_patterns: Dict[str, re.Pattern] = {}
    
    @abstractmethod
    def parse_line(self, line: str) -> Optional[InferenceLogEntry]:
        """
        Parse a single log line.
        
        Args:
            line: Raw log line string
            
        Returns:
            InferenceLogEntry if successfully parsed, None otherwise
        """
        pass
    
    @abstractmethod
    def detect(self, line: str) -> bool:
        """
        Detect if this parser can handle the given log format.
        
        Args:
            line: Sample log line
            
        Returns:
            True if this parser can parse the format
        """
        pass
    
    def parse_file(self, filepath: str) -> Iterator[InferenceLogEntry]:
        """
        Parse all entries from a log file.
        
        Args:
            filepath: Path to the log file
            
        Yields:
            InferenceLogEntry for each successfully parsed line
        """
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = self.parse_line(line.strip())
                    if entry:
                        yield entry
                except Exception as e:
                    # Log parsing error but continue
                    pass
    
    def parse_lines(self, lines: List[str]) -> List[InferenceLogEntry]:
        """
        Parse multiple log lines.
        
        Args:
            lines: List of log lines
            
        Returns:
            List of successfully parsed entries
        """
        entries = []
        for line in lines:
            entry = self.parse_line(line.strip())
            if entry:
                entries.append(entry)
        return entries
    
    def _compile_pattern(self, name: str, pattern: str) -> re.Pattern:
        """Compile and cache regex pattern"""
        if name not in self._compiled_patterns:
            self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)
        return self._compiled_patterns[name]
    
    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse timestamp from various formats"""
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%d/%b/%Y:%H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(ts_str.strip(), fmt)
            except ValueError:
                continue
        
        # Try parsing Unix timestamp
        try:
            ts_float = float(ts_str)
            return datetime.fromtimestamp(ts_float)
        except (ValueError, OSError):
            pass
        
        return None
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert to int"""
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
