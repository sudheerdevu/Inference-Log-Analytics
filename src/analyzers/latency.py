"""
Latency Analyzer Module

Provides latency analysis for inference logs.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..parsers.base import InferenceLogEntry


@dataclass
class LatencyStats:
    """Latency statistics container"""
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    p999: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'count': self.count,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'p50': self.p50,
            'p90': self.p90,
            'p95': self.p95,
            'p99': self.p99,
            'p999': self.p999,
        }


class LatencyAnalyzer:
    """Analyzes inference latency patterns"""
    
    def __init__(self, entries: Optional[List[InferenceLogEntry]] = None):
        self.entries = entries or []
    
    def add_entries(self, entries: List[InferenceLogEntry]) -> None:
        """Add entries for analysis"""
        self.entries.extend(entries)
    
    def get_latencies(self, 
                      model_name: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      success_only: bool = True) -> List[float]:
        """
        Get list of latency values with optional filtering.
        
        Args:
            model_name: Filter by model name
            start_time: Filter by start time
            end_time: Filter by end time
            success_only: Only include successful requests
            
        Returns:
            List of latency values in milliseconds
        """
        latencies = []
        for entry in self.entries:
            if model_name and entry.model_name != model_name:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if success_only and not entry.is_success:
                continue
            latencies.append(entry.latency_ms)
        return latencies
    
    def compute_stats(self, 
                      model_name: Optional[str] = None,
                      **filters) -> LatencyStats:
        """
        Compute latency statistics.
        
        Args:
            model_name: Filter by model name
            **filters: Additional filters passed to get_latencies
            
        Returns:
            LatencyStats object with computed metrics
        """
        latencies = self.get_latencies(model_name=model_name, **filters)
        
        if not latencies:
            return LatencyStats(
                count=0, mean=0, median=0, std=0, min=0, max=0,
                p50=0, p90=0, p95=0, p99=0, p999=0
            )
        
        arr = np.array(latencies)
        
        return LatencyStats(
            count=len(arr),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            p999=float(np.percentile(arr, 99.9)),
        )
    
    def compute_stats_by_model(self) -> Dict[str, LatencyStats]:
        """Compute latency stats for each model"""
        models = set(e.model_name for e in self.entries)
        return {model: self.compute_stats(model_name=model) for model in models}
    
    def compute_time_series(self,
                            interval: timedelta = timedelta(minutes=1),
                            model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Compute latency statistics over time intervals.
        
        Args:
            interval: Time interval for bucketing
            model_name: Optional model filter
            
        Returns:
            List of dicts with timestamp and latency stats
        """
        latencies = [(e.timestamp, e.latency_ms) for e in self.entries
                     if (model_name is None or e.model_name == model_name) and e.is_success]
        
        if not latencies:
            return []
        
        latencies.sort(key=lambda x: x[0])
        
        result = []
        current_bucket_start = latencies[0][0].replace(second=0, microsecond=0)
        current_bucket = []
        
        for ts, lat in latencies:
            while ts >= current_bucket_start + interval:
                if current_bucket:
                    arr = np.array(current_bucket)
                    result.append({
                        'timestamp': current_bucket_start.isoformat(),
                        'count': len(arr),
                        'mean': float(np.mean(arr)),
                        'p50': float(np.percentile(arr, 50)),
                        'p95': float(np.percentile(arr, 95)),
                        'p99': float(np.percentile(arr, 99)),
                    })
                else:
                    result.append({
                        'timestamp': current_bucket_start.isoformat(),
                        'count': 0,
                        'mean': 0,
                        'p50': 0,
                        'p95': 0,
                        'p99': 0,
                    })
                current_bucket_start += interval
                current_bucket = []
            
            current_bucket.append(lat)
        
        # Handle last bucket
        if current_bucket:
            arr = np.array(current_bucket)
            result.append({
                'timestamp': current_bucket_start.isoformat(),
                'count': len(arr),
                'mean': float(np.mean(arr)),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
            })
        
        return result
    
    def compute_histogram(self,
                          bins: int = 50,
                          model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute latency histogram.
        
        Args:
            bins: Number of histogram bins
            model_name: Optional model filter
            
        Returns:
            Dict with bin edges and counts
        """
        latencies = self.get_latencies(model_name=model_name)
        
        if not latencies:
            return {'edges': [], 'counts': []}
        
        counts, edges = np.histogram(latencies, bins=bins)
        
        return {
            'edges': edges.tolist(),
            'counts': counts.tolist(),
        }
    
    def find_slow_requests(self,
                           threshold_percentile: float = 99,
                           model_name: Optional[str] = None) -> List[InferenceLogEntry]:
        """
        Find requests slower than the given percentile.
        
        Args:
            threshold_percentile: Percentile threshold (0-100)
            model_name: Optional model filter
            
        Returns:
            List of slow entries
        """
        latencies = self.get_latencies(model_name=model_name)
        
        if not latencies:
            return []
        
        threshold = np.percentile(latencies, threshold_percentile)
        
        return [e for e in self.entries
                if (model_name is None or e.model_name == model_name)
                and e.latency_ms >= threshold]
