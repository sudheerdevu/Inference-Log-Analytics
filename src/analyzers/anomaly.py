"""
Anomaly Detection Module

Provides anomaly detection for inference logs using statistical methods.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from ..parsers.base import InferenceLogEntry


class AnomalyType(Enum):
    """Types of anomalies"""
    LATENCY_SPIKE = 'latency_spike'
    LATENCY_DEGRADATION = 'latency_degradation'
    THROUGHPUT_DROP = 'throughput_drop'
    THROUGHPUT_SPIKE = 'throughput_spike'
    ERROR_RATE_SPIKE = 'error_rate_spike'
    PATTERN_CHANGE = 'pattern_change'


@dataclass
class Anomaly:
    """Anomaly detection result"""
    anomaly_type: AnomalyType
    timestamp: datetime
    severity: float  # 0-1 scale
    description: str
    affected_model: Optional[str]
    metric_value: float
    expected_value: float
    z_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'anomaly_type': self.anomaly_type.value,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'description': self.description,
            'affected_model': self.affected_model,
            'metric_value': self.metric_value,
            'expected_value': self.expected_value,
            'z_score': self.z_score,
            'metadata': self.metadata,
        }


class MovingStats:
    """Maintains moving statistics for streaming anomaly detection"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def add(self, value: float) -> None:
        """Add a value to the window"""
        if len(self.values) == self.window_size:
            old = self.values[0]
            self._sum -= old
            self._sum_sq -= old * old
        
        self.values.append(value)
        self._sum += value
        self._sum_sq += value * value
    
    @property
    def mean(self) -> float:
        """Get current mean"""
        n = len(self.values)
        return self._sum / n if n > 0 else 0.0
    
    @property
    def std(self) -> float:
        """Get current standard deviation"""
        n = len(self.values)
        if n < 2:
            return 0.0
        variance = (self._sum_sq / n) - (self._sum / n) ** 2
        return np.sqrt(max(0, variance))
    
    def z_score(self, value: float) -> float:
        """Calculate z-score for a value"""
        std = self.std
        if std == 0:
            return 0.0
        return (value - self.mean) / std


class AnomalyDetector:
    """Detects anomalies in inference logs"""
    
    def __init__(self,
                 entries: Optional[List[InferenceLogEntry]] = None,
                 z_score_threshold: float = 3.0,
                 window_size: int = 100):
        self.entries = entries or []
        self.z_score_threshold = z_score_threshold
        self.window_size = window_size
    
    def add_entries(self, entries: List[InferenceLogEntry]) -> None:
        """Add entries for analysis"""
        self.entries.extend(entries)
    
    def detect_latency_anomalies(self,
                                 model_name: Optional[str] = None,
                                 threshold: Optional[float] = None) -> List[Anomaly]:
        """
        Detect latency anomalies using z-score method.
        
        Args:
            model_name: Optional model filter
            threshold: Custom z-score threshold
            
        Returns:
            List of detected anomalies
        """
        threshold = threshold or self.z_score_threshold
        
        filtered = [e for e in self.entries
                    if (model_name is None or e.model_name == model_name)
                    and e.is_success]
        
        if len(filtered) < self.window_size:
            return []
        
        filtered.sort(key=lambda x: x.timestamp)
        
        anomalies = []
        stats = MovingStats(window_size=self.window_size)
        
        # Initialize with first window
        for entry in filtered[:self.window_size]:
            stats.add(entry.latency_ms)
        
        # Detect anomalies
        for entry in filtered[self.window_size:]:
            z = stats.z_score(entry.latency_ms)
            
            if abs(z) > threshold:
                anomaly_type = AnomalyType.LATENCY_SPIKE if z > 0 else AnomalyType.LATENCY_DEGRADATION
                severity = min(1.0, abs(z) / (2 * threshold))
                
                anomalies.append(Anomaly(
                    anomaly_type=anomaly_type,
                    timestamp=entry.timestamp,
                    severity=severity,
                    description=f"Latency {anomaly_type.value}: {entry.latency_ms:.2f}ms (expected ~{stats.mean:.2f}ms)",
                    affected_model=entry.model_name,
                    metric_value=entry.latency_ms,
                    expected_value=stats.mean,
                    z_score=z,
                    metadata={
                        'request_id': entry.request_id,
                        'std': stats.std,
                    }
                ))
            
            stats.add(entry.latency_ms)
        
        return anomalies
    
    def detect_throughput_anomalies(self,
                                    interval: timedelta = timedelta(minutes=1),
                                    model_name: Optional[str] = None,
                                    threshold: Optional[float] = None) -> List[Anomaly]:
        """
        Detect throughput anomalies.
        
        Args:
            interval: Time interval for bucketing
            model_name: Optional model filter
            threshold: Custom z-score threshold
            
        Returns:
            List of detected anomalies
        """
        threshold = threshold or self.z_score_threshold
        
        filtered = [e for e in self.entries
                    if model_name is None or e.model_name == model_name]
        
        if not filtered:
            return []
        
        filtered.sort(key=lambda x: x.timestamp)
        
        # Bucket entries by time
        buckets = []
        current_bucket_start = filtered[0].timestamp.replace(second=0, microsecond=0)
        current_count = 0
        
        for entry in filtered:
            while entry.timestamp >= current_bucket_start + interval:
                buckets.append((current_bucket_start, current_count))
                current_bucket_start += interval
                current_count = 0
            current_count += 1
        
        buckets.append((current_bucket_start, current_count))
        
        if len(buckets) < self.window_size:
            return []
        
        anomalies = []
        stats = MovingStats(window_size=self.window_size)
        
        # Initialize with first window
        for _, count in buckets[:self.window_size]:
            stats.add(count)
        
        # Detect anomalies
        for ts, count in buckets[self.window_size:]:
            z = stats.z_score(count)
            
            if abs(z) > threshold:
                anomaly_type = AnomalyType.THROUGHPUT_SPIKE if z > 0 else AnomalyType.THROUGHPUT_DROP
                severity = min(1.0, abs(z) / (2 * threshold))
                
                anomalies.append(Anomaly(
                    anomaly_type=anomaly_type,
                    timestamp=ts,
                    severity=severity,
                    description=f"Throughput {anomaly_type.value}: {count} requests (expected ~{stats.mean:.1f})",
                    affected_model=model_name,
                    metric_value=count,
                    expected_value=stats.mean,
                    z_score=z,
                    metadata={
                        'interval_minutes': interval.total_seconds() / 60,
                        'std': stats.std,
                    }
                ))
            
            stats.add(count)
        
        return anomalies
    
    def detect_error_rate_anomalies(self,
                                    interval: timedelta = timedelta(minutes=5),
                                    model_name: Optional[str] = None,
                                    threshold: Optional[float] = None,
                                    min_requests: int = 10) -> List[Anomaly]:
        """
        Detect error rate anomalies.
        
        Args:
            interval: Time interval for bucketing
            model_name: Optional model filter
            threshold: Custom z-score threshold
            min_requests: Minimum requests in bucket to analyze
            
        Returns:
            List of detected anomalies
        """
        threshold = threshold or self.z_score_threshold
        
        filtered = [e for e in self.entries
                    if model_name is None or e.model_name == model_name]
        
        if not filtered:
            return []
        
        filtered.sort(key=lambda x: x.timestamp)
        
        # Bucket entries by time with error rates
        buckets = []
        current_bucket_start = filtered[0].timestamp.replace(second=0, microsecond=0)
        current_total = 0
        current_errors = 0
        
        for entry in filtered:
            while entry.timestamp >= current_bucket_start + interval:
                if current_total >= min_requests:
                    buckets.append((current_bucket_start, current_errors / current_total, current_total))
                current_bucket_start += interval
                current_total = 0
                current_errors = 0
            
            current_total += 1
            if not entry.is_success:
                current_errors += 1
        
        if current_total >= min_requests:
            buckets.append((current_bucket_start, current_errors / current_total, current_total))
        
        if len(buckets) < self.window_size:
            return []
        
        anomalies = []
        stats = MovingStats(window_size=self.window_size)
        
        # Initialize with first window
        for _, rate, _ in buckets[:self.window_size]:
            stats.add(rate)
        
        # Detect anomalies
        for ts, rate, total in buckets[self.window_size:]:
            z = stats.z_score(rate)
            
            if z > threshold:  # Only spikes, not drops
                severity = min(1.0, z / (2 * threshold))
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.ERROR_RATE_SPIKE,
                    timestamp=ts,
                    severity=severity,
                    description=f"Error rate spike: {rate*100:.1f}% (expected ~{stats.mean*100:.1f}%)",
                    affected_model=model_name,
                    metric_value=rate,
                    expected_value=stats.mean,
                    z_score=z,
                    metadata={
                        'total_requests': total,
                        'error_count': int(rate * total),
                        'std': stats.std,
                    }
                ))
            
            stats.add(rate)
        
        return anomalies
    
    def detect_all_anomalies(self,
                             model_name: Optional[str] = None,
                             interval: timedelta = timedelta(minutes=1)) -> List[Anomaly]:
        """
        Run all anomaly detectors.
        
        Args:
            model_name: Optional model filter
            interval: Time interval for bucketing
            
        Returns:
            List of all detected anomalies, sorted by timestamp
        """
        anomalies = []
        
        anomalies.extend(self.detect_latency_anomalies(model_name=model_name))
        anomalies.extend(self.detect_throughput_anomalies(interval=interval, model_name=model_name))
        anomalies.extend(self.detect_error_rate_anomalies(interval=interval, model_name=model_name))
        
        # Sort by timestamp
        anomalies.sort(key=lambda a: a.timestamp)
        
        return anomalies
    
    def detect_anomalies_by_model(self,
                                  interval: timedelta = timedelta(minutes=1)) -> Dict[str, List[Anomaly]]:
        """
        Detect anomalies for each model separately.
        
        Args:
            interval: Time interval for bucketing
            
        Returns:
            Dict mapping model name to anomaly list
        """
        models = set(e.model_name for e in self.entries)
        return {
            model: self.detect_all_anomalies(model_name=model, interval=interval)
            for model in models
        }
    
    def get_anomaly_summary(self,
                            model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of detected anomalies.
        
        Args:
            model_name: Optional model filter
            
        Returns:
            Summary dict with counts and severity breakdown
        """
        anomalies = self.detect_all_anomalies(model_name=model_name)
        
        if not anomalies:
            return {
                'total_anomalies': 0,
                'by_type': {},
                'by_severity': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                'affected_models': [],
            }
        
        by_type = {}
        by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        affected_models = set()
        
        for a in anomalies:
            # Count by type
            type_name = a.anomaly_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            
            # Count by severity
            if a.severity < 0.25:
                by_severity['low'] += 1
            elif a.severity < 0.5:
                by_severity['medium'] += 1
            elif a.severity < 0.75:
                by_severity['high'] += 1
            else:
                by_severity['critical'] += 1
            
            if a.affected_model:
                affected_models.add(a.affected_model)
        
        return {
            'total_anomalies': len(anomalies),
            'by_type': by_type,
            'by_severity': by_severity,
            'affected_models': list(affected_models),
        }


class SeasonalAnomalyDetector:
    """Detects anomalies accounting for seasonal patterns"""
    
    def __init__(self, 
                 entries: Optional[List[InferenceLogEntry]] = None,
                 period_hours: int = 24,
                 z_score_threshold: float = 3.0):
        self.entries = entries or []
        self.period_hours = period_hours
        self.z_score_threshold = z_score_threshold
    
    def add_entries(self, entries: List[InferenceLogEntry]) -> None:
        """Add entries for analysis"""
        self.entries.extend(entries)
    
    def detect_latency_anomalies_seasonal(self,
                                          model_name: Optional[str] = None) -> List[Anomaly]:
        """
        Detect latency anomalies accounting for daily patterns.
        """
        filtered = [e for e in self.entries
                    if (model_name is None or e.model_name == model_name)
                    and e.is_success]
        
        if not filtered:
            return []
        
        # Group by hour of day
        hourly_latencies = {h: [] for h in range(24)}
        for entry in filtered:
            hour = entry.timestamp.hour
            hourly_latencies[hour].append((entry, entry.latency_ms))
        
        # Compute hourly baselines
        hourly_stats = {}
        for hour, entries in hourly_latencies.items():
            if entries:
                lats = [l for _, l in entries]
                hourly_stats[hour] = {
                    'mean': np.mean(lats),
                    'std': np.std(lats),
                }
        
        # Detect anomalies against seasonal baseline
        anomalies = []
        for hour, entries in hourly_latencies.items():
            if hour not in hourly_stats:
                continue
            
            stats = hourly_stats[hour]
            if stats['std'] == 0:
                continue
            
            for entry, latency in entries:
                z = (latency - stats['mean']) / stats['std']
                
                if abs(z) > self.z_score_threshold:
                    anomaly_type = AnomalyType.LATENCY_SPIKE if z > 0 else AnomalyType.LATENCY_DEGRADATION
                    severity = min(1.0, abs(z) / (2 * self.z_score_threshold))
                    
                    anomalies.append(Anomaly(
                        anomaly_type=anomaly_type,
                        timestamp=entry.timestamp,
                        severity=severity,
                        description=f"Seasonal latency anomaly at hour {hour}: {latency:.2f}ms (expected ~{stats['mean']:.2f}ms for this hour)",
                        affected_model=entry.model_name,
                        metric_value=latency,
                        expected_value=stats['mean'],
                        z_score=z,
                        metadata={
                            'hour': hour,
                            'hourly_std': stats['std'],
                        }
                    ))
        
        anomalies.sort(key=lambda a: a.timestamp)
        return anomalies
