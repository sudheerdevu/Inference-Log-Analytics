"""
Throughput Analyzer Module

Provides throughput and request rate analysis for inference logs.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from ..parsers.base import InferenceLogEntry


@dataclass
class ThroughputStats:
    """Throughput statistics container"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    requests_per_second: float
    duration_seconds: float
    peak_rps: float
    average_batch_size: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'requests_per_second': self.requests_per_second,
            'duration_seconds': self.duration_seconds,
            'peak_rps': self.peak_rps,
            'average_batch_size': self.average_batch_size,
        }


class ThroughputAnalyzer:
    """Analyzes inference throughput and request rates"""
    
    def __init__(self, entries: Optional[List[InferenceLogEntry]] = None):
        self.entries = entries or []
    
    def add_entries(self, entries: List[InferenceLogEntry]) -> None:
        """Add entries for analysis"""
        self.entries.extend(entries)
    
    def get_time_range(self, 
                       model_name: Optional[str] = None) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the time range of entries.
        
        Args:
            model_name: Optional model filter
            
        Returns:
            Tuple of (start_time, end_time)
        """
        filtered = [e.timestamp for e in self.entries
                    if model_name is None or e.model_name == model_name]
        
        if not filtered:
            return None, None
        
        return min(filtered), max(filtered)
    
    def compute_stats(self, 
                      model_name: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> ThroughputStats:
        """
        Compute throughput statistics.
        
        Args:
            model_name: Filter by model name
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            ThroughputStats object with computed metrics
        """
        filtered = []
        for entry in self.entries:
            if model_name and entry.model_name != model_name:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            filtered.append(entry)
        
        if not filtered:
            return ThroughputStats(
                total_requests=0, successful_requests=0, failed_requests=0,
                success_rate=0, requests_per_second=0, duration_seconds=0,
                peak_rps=0, average_batch_size=0
            )
        
        total = len(filtered)
        successful = sum(1 for e in filtered if e.is_success)
        failed = total - successful
        
        timestamps = sorted(e.timestamp for e in filtered)
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        
        if duration > 0:
            rps = total / duration
        else:
            rps = total  # All in same second
        
        # Calculate peak RPS using 1-second buckets
        peak_rps = self._calculate_peak_rps(filtered, interval_seconds=1)
        
        # Calculate average batch size
        batch_sizes = [e.batch_size for e in filtered if e.batch_size is not None]
        avg_batch = np.mean(batch_sizes) if batch_sizes else 1.0
        
        return ThroughputStats(
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            success_rate=successful / total if total > 0 else 0,
            requests_per_second=rps,
            duration_seconds=duration,
            peak_rps=peak_rps,
            average_batch_size=float(avg_batch),
        )
    
    def _calculate_peak_rps(self, 
                            entries: List[InferenceLogEntry],
                            interval_seconds: int = 1) -> float:
        """Calculate peak requests per second"""
        if not entries:
            return 0
        
        # Group by time bucket
        buckets = defaultdict(int)
        for entry in entries:
            bucket = entry.timestamp.replace(microsecond=0)
            buckets[bucket] += 1
        
        if not buckets:
            return 0
        
        return max(buckets.values()) / interval_seconds
    
    def compute_stats_by_model(self) -> Dict[str, ThroughputStats]:
        """Compute throughput stats for each model"""
        models = set(e.model_name for e in self.entries)
        return {model: self.compute_stats(model_name=model) for model in models}
    
    def compute_time_series(self,
                            interval: timedelta = timedelta(minutes=1),
                            model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Compute throughput statistics over time intervals.
        
        Args:
            interval: Time interval for bucketing
            model_name: Optional model filter
            
        Returns:
            List of dicts with timestamp and throughput stats
        """
        filtered = [e for e in self.entries
                    if model_name is None or e.model_name == model_name]
        
        if not filtered:
            return []
        
        filtered.sort(key=lambda x: x.timestamp)
        
        result = []
        interval_seconds = interval.total_seconds()
        current_bucket_start = filtered[0].timestamp.replace(second=0, microsecond=0)
        current_bucket = []
        
        for entry in filtered:
            while entry.timestamp >= current_bucket_start + interval:
                total = len(current_bucket)
                successful = sum(1 for e in current_bucket if e.is_success)
                
                result.append({
                    'timestamp': current_bucket_start.isoformat(),
                    'total_requests': total,
                    'successful_requests': successful,
                    'failed_requests': total - successful,
                    'rps': total / interval_seconds if interval_seconds > 0 else 0,
                    'success_rate': successful / total if total > 0 else 0,
                })
                
                current_bucket_start += interval
                current_bucket = []
            
            current_bucket.append(entry)
        
        # Handle last bucket
        if current_bucket:
            total = len(current_bucket)
            successful = sum(1 for e in current_bucket if e.is_success)
            
            result.append({
                'timestamp': current_bucket_start.isoformat(),
                'total_requests': total,
                'successful_requests': successful,
                'failed_requests': total - successful,
                'rps': total / interval_seconds if interval_seconds > 0 else 0,
                'success_rate': successful / total if total > 0 else 0,
            })
        
        return result
    
    def compute_daily_breakdown(self, 
                                model_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compute throughput breakdown by day of week.
        
        Args:
            model_name: Optional model filter
            
        Returns:
            Dict mapping day name to stats
        """
        filtered = [e for e in self.entries
                    if model_name is None or e.model_name == model_name]
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        breakdown = {day: [] for day in days}
        
        for entry in filtered:
            day_name = days[entry.timestamp.weekday()]
            breakdown[day_name].append(entry)
        
        result = {}
        for day, entries in breakdown.items():
            total = len(entries)
            successful = sum(1 for e in entries if e.is_success)
            result[day] = {
                'total_requests': total,
                'successful_requests': successful,
                'success_rate': successful / total if total > 0 else 0,
            }
        
        return result
    
    def compute_hourly_breakdown(self,
                                 model_name: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
        """
        Compute throughput breakdown by hour of day.
        
        Args:
            model_name: Optional model filter
            
        Returns:
            Dict mapping hour (0-23) to stats
        """
        filtered = [e for e in self.entries
                    if model_name is None or e.model_name == model_name]
        
        breakdown = {hour: [] for hour in range(24)}
        
        for entry in filtered:
            hour = entry.timestamp.hour
            breakdown[hour].append(entry)
        
        result = {}
        for hour, entries in breakdown.items():
            total = len(entries)
            successful = sum(1 for e in entries if e.is_success)
            
            latencies = [e.latency_ms for e in entries if e.is_success]
            avg_latency = np.mean(latencies) if latencies else 0
            
            result[hour] = {
                'total_requests': total,
                'successful_requests': successful,
                'success_rate': successful / total if total > 0 else 0,
                'avg_latency_ms': float(avg_latency),
            }
        
        return result
    
    def find_throughput_anomalies(self,
                                  threshold_factor: float = 2.0,
                                  interval: timedelta = timedelta(minutes=5),
                                  model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find time periods with anomalous throughput.
        
        Args:
            threshold_factor: Factor above/below mean to consider anomalous
            interval: Time interval for analysis
            model_name: Optional model filter
            
        Returns:
            List of anomalous time periods
        """
        time_series = self.compute_time_series(interval=interval, model_name=model_name)
        
        if len(time_series) < 3:
            return []
        
        request_counts = [ts['total_requests'] for ts in time_series]
        mean_count = np.mean(request_counts)
        std_count = np.std(request_counts)
        
        anomalies = []
        for ts in time_series:
            count = ts['total_requests']
            if std_count > 0:
                z_score = (count - mean_count) / std_count
                if abs(z_score) > threshold_factor:
                    anomalies.append({
                        'timestamp': ts['timestamp'],
                        'request_count': count,
                        'expected_count': mean_count,
                        'z_score': z_score,
                        'anomaly_type': 'spike' if z_score > 0 else 'drop',
                    })
        
        return anomalies
