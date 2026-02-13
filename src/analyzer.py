#!/usr/bin/env python3
"""
Inference Log Analytics Engine - Main Module

Production-grade toolkit for analyzing AI inference logs.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import statistics

import numpy as np
import pandas as pd


@dataclass
class InferenceRecord:
    """Single inference record."""
    timestamp: datetime
    model: str
    latency_ms: float
    queue_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    batch_size: int = 1
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics."""
    count: int
    min: float
    max: float
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    p999: float
    
    @classmethod
    def from_values(cls, latencies: List[float]) -> 'LatencyStats':
        """Compute stats from list of latencies."""
        if not latencies:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        arr = np.array(latencies)
        return cls(
            count=len(arr),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            p999=float(np.percentile(arr, 99.9))
        )


@dataclass
class ErrorStats:
    """Error statistics."""
    total_requests: int
    failed_requests: int
    rate: float
    error_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class ThroughputStats:
    """Throughput statistics."""
    total_inferences: int
    duration_seconds: float
    avg_throughput: float  # inferences/sec
    peak_throughput: float
    min_throughput: float


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    start_time: datetime
    end_time: datetime
    latency: LatencyStats
    errors: ErrorStats
    throughput: ThroughputStats
    models: Dict[str, LatencyStats]
    anomalies: List[Dict[str, Any]]


class LogParser(ABC):
    """Base class for log parsers."""
    
    @abstractmethod
    def parse(self, content: str) -> Iterator[InferenceRecord]:
        """Parse log content and yield inference records."""
        pass
    
    @classmethod
    def from_file(cls, path: str) -> Iterator[InferenceRecord]:
        """Parse from file."""
        with open(path, 'r') as f:
            yield from cls().parse(f.read())


class ONNXRuntimeParser(LogParser):
    """Parser for ONNX Runtime profiling output."""
    
    # Pattern for ONNX Runtime profiling JSON
    PROFILE_PATTERN = re.compile(
        r'\{"cat":"Session","name":"([^"]+)".*?"dur":(\d+)'
    )
    
    def parse(self, content: str) -> Iterator[InferenceRecord]:
        """Parse ONNX Runtime profile JSON."""
        try:
            data = json.loads(content)
            events = data if isinstance(data, list) else data.get('traceEvents', [])
            
            for event in events:
                if event.get('cat') == 'Session' and 'dur' in event:
                    yield InferenceRecord(
                        timestamp=datetime.fromtimestamp(
                            event.get('ts', 0) / 1_000_000
                        ),
                        model=event.get('args', {}).get('model', 'unknown'),
                        latency_ms=event.get('dur', 0) / 1000,  # us to ms
                        execution_time_ms=event.get('dur', 0) / 1000,
                        metadata=event.get('args', {})
                    )
        except json.JSONDecodeError:
            # Try line-by-line parsing
            for line in content.split('\n'):
                match = self.PROFILE_PATTERN.search(line)
                if match:
                    yield InferenceRecord(
                        timestamp=datetime.now(),
                        model=match.group(1),
                        latency_ms=int(match.group(2)) / 1000
                    )


class TritonParser(LogParser):
    """Parser for NVIDIA Triton Inference Server logs."""
    
    # Pattern: timestamp model_name latency_us
    LOG_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*?'
        r'model=(\w+).*?latency=(\d+)'
    )
    
    def parse(self, content: str) -> Iterator[InferenceRecord]:
        """Parse Triton log lines."""
        for line in content.split('\n'):
            match = self.LOG_PATTERN.search(line)
            if match:
                yield InferenceRecord(
                    timestamp=datetime.fromisoformat(match.group(1)),
                    model=match.group(2),
                    latency_ms=int(match.group(3)) / 1000
                )


class CustomJSONParser(LogParser):
    """Parser for custom JSON log format."""
    
    def __init__(self, 
                 timestamp_field: str = 'timestamp',
                 latency_field: str = 'latency_ms',
                 model_field: str = 'model'):
        self.timestamp_field = timestamp_field
        self.latency_field = latency_field
        self.model_field = model_field
    
    def parse(self, content: str) -> Iterator[InferenceRecord]:
        """Parse JSON lines."""
        for line in content.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                yield InferenceRecord(
                    timestamp=datetime.fromisoformat(
                        str(data.get(self.timestamp_field, ''))
                    ) if self.timestamp_field in data else datetime.now(),
                    model=data.get(self.model_field, 'unknown'),
                    latency_ms=float(data.get(self.latency_field, 0)),
                    batch_size=data.get('batch_size', 1),
                    status=data.get('status', 'success'),
                    error_message=data.get('error'),
                    metadata=data
                )
            except (json.JSONDecodeError, ValueError):
                continue


class AnomalyDetector:
    """Detect anomalies in inference patterns."""
    
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
    
    def detect(self, records: List[InferenceRecord]) -> List[Dict[str, Any]]:
        """Detect anomalous records."""
        if len(records) < 10:
            return []
        
        latencies = [r.latency_ms for r in records]
        mean = statistics.mean(latencies)
        std = statistics.stdev(latencies)
        
        if std == 0:
            return []
        
        anomalies = []
        for i, record in enumerate(records):
            z_score = (record.latency_ms - mean) / std
            if abs(z_score) > self.z_threshold:
                anomalies.append({
                    'index': i,
                    'timestamp': record.timestamp,
                    'latency_ms': record.latency_ms,
                    'z_score': z_score,
                    'type': 'high_latency' if z_score > 0 else 'low_latency'
                })
        
        return anomalies


class LogAnalyzer:
    """Main analysis class."""
    
    PARSERS = {
        'onnx_runtime': ONNXRuntimeParser,
        'triton': TritonParser,
        'json': CustomJSONParser,
    }
    
    def __init__(self, records: List[InferenceRecord]):
        self.records = records
        self.anomaly_detector = AnomalyDetector()
    
    @classmethod
    def from_file(cls, path: str, format: str = 'json') -> 'LogAnalyzer':
        """Load and parse from file."""
        parser_cls = cls.PARSERS.get(format, CustomJSONParser)
        parser = parser_cls()
        
        with open(path, 'r') as f:
            records = list(parser.parse(f.read()))
        
        return cls(records)
    
    @classmethod
    def from_records(cls, records: List[InferenceRecord]) -> 'LogAnalyzer':
        """Create from existing records."""
        return cls(records)
    
    def analyze(self) -> AnalysisReport:
        """Perform full analysis."""
        if not self.records:
            raise ValueError("No records to analyze")
        
        # Time range
        timestamps = [r.timestamp for r in self.records]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        # Latency analysis
        latencies = [r.latency_ms for r in self.records]
        latency_stats = LatencyStats.from_values(latencies)
        
        # Error analysis
        total = len(self.records)
        failed = [r for r in self.records if r.status != 'success']
        error_types: Dict[str, int] = {}
        for r in failed:
            err = r.error_message or 'unknown'
            error_types[err] = error_types.get(err, 0) + 1
        
        error_stats = ErrorStats(
            total_requests=total,
            failed_requests=len(failed),
            rate=len(failed) / total if total > 0 else 0,
            error_types=error_types
        )
        
        # Throughput analysis
        duration = (end_time - start_time).total_seconds()
        throughput_stats = ThroughputStats(
            total_inferences=total,
            duration_seconds=duration,
            avg_throughput=total / duration if duration > 0 else 0,
            peak_throughput=0,  # Would need time-series buckets
            min_throughput=0
        )
        
        # Per-model analysis
        models: Dict[str, LatencyStats] = {}
        model_records: Dict[str, List[float]] = {}
        for r in self.records:
            if r.model not in model_records:
                model_records[r.model] = []
            model_records[r.model].append(r.latency_ms)
        
        for model, lats in model_records.items():
            models[model] = LatencyStats.from_values(lats)
        
        # Anomaly detection
        anomalies = self.anomaly_detector.detect(self.records)
        
        return AnalysisReport(
            start_time=start_time,
            end_time=end_time,
            latency=latency_stats,
            errors=error_stats,
            throughput=throughput_stats,
            models=models,
            anomalies=anomalies
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to pandas DataFrame."""
        return pd.DataFrame([
            {
                'timestamp': r.timestamp,
                'model': r.model,
                'latency_ms': r.latency_ms,
                'queue_time_ms': r.queue_time_ms,
                'execution_time_ms': r.execution_time_ms,
                'batch_size': r.batch_size,
                'status': r.status,
            }
            for r in self.records
        ])
    
    def generate_report(self) -> str:
        """Generate text report."""
        report = self.analyze()
        
        lines = [
            "=" * 60,
            "         Inference Log Analysis Report",
            "=" * 60,
            "",
            f"Time Range: {report.start_time} to {report.end_time}",
            f"Total Inferences: {report.latency.count:,}",
            "",
            "LATENCY STATISTICS",
            "-" * 40,
            f"  Min:    {report.latency.min:.2f} ms",
            f"  Max:    {report.latency.max:.2f} ms",
            f"  Mean:   {report.latency.mean:.2f} ms",
            f"  StdDev: {report.latency.std:.2f} ms",
            "",
            "  Percentiles:",
            f"    P50:   {report.latency.p50:.2f} ms",
            f"    P90:   {report.latency.p90:.2f} ms",
            f"    P95:   {report.latency.p95:.2f} ms",
            f"    P99:   {report.latency.p99:.2f} ms",
            f"    P99.9: {report.latency.p999:.2f} ms",
            "",
            "ERROR STATISTICS",
            "-" * 40,
            f"  Total Requests: {report.errors.total_requests:,}",
            f"  Failed:         {report.errors.failed_requests:,}",
            f"  Error Rate:     {report.errors.rate:.2%}",
            "",
            "THROUGHPUT",
            "-" * 40,
            f"  Duration:    {report.throughput.duration_seconds:.1f} seconds",
            f"  Avg QPS:     {report.throughput.avg_throughput:.2f}",
            "",
            "ANOMALIES DETECTED",
            "-" * 40,
            f"  Count: {len(report.anomalies)}",
        ]
        
        if report.anomalies:
            for a in report.anomalies[:5]:
                lines.append(f"    - {a['timestamp']}: {a['latency_ms']:.2f}ms (z={a['z_score']:.2f})")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)


# CLI entry point
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze inference logs')
    parser.add_argument('file', help='Log file to analyze')
    parser.add_argument('--format', choices=['onnx_runtime', 'triton', 'json'],
                       default='json', help='Log format')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer.from_file(args.file, format=args.format)
    report = analyzer.generate_report()
    
    print(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
