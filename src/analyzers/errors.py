"""
Error Analyzer Module

Provides error analysis and categorization for inference logs.
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from ..parsers.base import InferenceLogEntry


@dataclass
class ErrorCategory:
    """Error category definition"""
    name: str
    patterns: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str


# Default error categories
DEFAULT_ERROR_CATEGORIES = [
    ErrorCategory(
        name='timeout',
        patterns=[r'timeout', r'timed?\s*out', r'deadline\s*exceeded'],
        severity='high',
        description='Request timeout errors'
    ),
    ErrorCategory(
        name='memory',
        patterns=[r'out\s*of\s*memory', r'oom', r'memory\s*allocation', r'cuda\s*out\s*of\s*memory'],
        severity='critical',
        description='Memory allocation failures'
    ),
    ErrorCategory(
        name='model_not_found',
        patterns=[r'model\s*not\s*found', r'no\s*such\s*model', r'unknown\s*model'],
        severity='medium',
        description='Model not found errors'
    ),
    ErrorCategory(
        name='input_validation',
        patterns=[r'invalid\s*input', r'validation\s*error', r'malformed', r'bad\s*request'],
        severity='low',
        description='Input validation failures'
    ),
    ErrorCategory(
        name='tensor_shape',
        patterns=[r'shape\s*mismatch', r'dimension\s*mismatch', r'incompatible\s*shape'],
        severity='medium',
        description='Tensor shape mismatches'
    ),
    ErrorCategory(
        name='gpu_error',
        patterns=[r'cuda\s*error', r'gpu\s*error', r'rocm\s*error', r'device\s*error'],
        severity='critical',
        description='GPU/device errors'
    ),
    ErrorCategory(
        name='network',
        patterns=[r'connection\s*refused', r'connection\s*reset', r'network\s*error'],
        severity='high',
        description='Network connectivity issues'
    ),
    ErrorCategory(
        name='rate_limit',
        patterns=[r'rate\s*limit', r'throttl', r'too\s*many\s*requests', r'429'],
        severity='medium',
        description='Rate limiting errors'
    ),
    ErrorCategory(
        name='internal',
        patterns=[r'internal\s*error', r'500', r'server\s*error'],
        severity='high',
        description='Internal server errors'
    ),
]


class ErrorAnalyzer:
    """Analyzes inference errors and failure patterns"""
    
    def __init__(self, 
                 entries: Optional[List[InferenceLogEntry]] = None,
                 categories: Optional[List[ErrorCategory]] = None):
        self.entries = entries or []
        self.categories = categories or DEFAULT_ERROR_CATEGORIES
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for efficiency"""
        compiled = {}
        for cat in self.categories:
            compiled[cat.name] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in cat.patterns
            ]
        return compiled
    
    def add_entries(self, entries: List[InferenceLogEntry]) -> None:
        """Add entries for analysis"""
        self.entries.extend(entries)
    
    def add_category(self, category: ErrorCategory) -> None:
        """Add a custom error category"""
        self.categories.append(category)
        self._compiled_patterns[category.name] = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in category.patterns
        ]
    
    def get_failed_entries(self,
                           model_name: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[InferenceLogEntry]:
        """Get all failed entries with optional filtering"""
        failed = []
        for entry in self.entries:
            if entry.is_success:
                continue
            if model_name and entry.model_name != model_name:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            failed.append(entry)
        return failed
    
    def categorize_error(self, error_message: str) -> Optional[str]:
        """
        Categorize an error message.
        
        Args:
            error_message: The error message to categorize
            
        Returns:
            Category name or None if uncategorized
        """
        if not error_message:
            return None
        
        for cat_name, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(error_message):
                    return cat_name
        
        return None
    
    def compute_error_stats(self,
                            model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute error statistics.
        
        Args:
            model_name: Optional model filter
            
        Returns:
            Dict with error statistics
        """
        filtered = [e for e in self.entries
                    if model_name is None or e.model_name == model_name]
        
        total = len(filtered)
        failed = [e for e in filtered if not e.is_success]
        
        if not filtered:
            return {
                'total_requests': 0,
                'failed_requests': 0,
                'error_rate': 0,
                'categories': {},
            }
        
        # Categorize errors
        category_counts = defaultdict(int)
        uncategorized = 0
        
        for entry in failed:
            category = self.categorize_error(entry.error_message or '')
            if category:
                category_counts[category] += 1
            else:
                uncategorized += 1
        
        return {
            'total_requests': total,
            'failed_requests': len(failed),
            'error_rate': len(failed) / total if total > 0 else 0,
            'categories': dict(category_counts),
            'uncategorized_errors': uncategorized,
        }
    
    def compute_error_breakdown_by_category(self,
                                            model_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compute detailed breakdown by error category.
        
        Args:
            model_name: Optional model filter
            
        Returns:
            Dict mapping category to detailed stats
        """
        failed = self.get_failed_entries(model_name=model_name)
        
        breakdown = {}
        for cat in self.categories:
            cat_entries = [e for e in failed 
                          if self.categorize_error(e.error_message or '') == cat.name]
            
            if cat_entries:
                breakdown[cat.name] = {
                    'count': len(cat_entries),
                    'severity': cat.severity,
                    'description': cat.description,
                    'sample_errors': [e.error_message for e in cat_entries[:5]],
                    'affected_models': list(set(e.model_name for e in cat_entries)),
                }
        
        return breakdown
    
    def compute_error_time_series(self,
                                  interval: timedelta = timedelta(minutes=5),
                                  model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Compute error rates over time.
        
        Args:
            interval: Time interval for bucketing
            model_name: Optional model filter
            
        Returns:
            List of dicts with timestamp and error stats
        """
        filtered = [e for e in self.entries
                    if model_name is None or e.model_name == model_name]
        
        if not filtered:
            return []
        
        filtered.sort(key=lambda x: x.timestamp)
        
        result = []
        current_bucket_start = filtered[0].timestamp.replace(second=0, microsecond=0)
        current_bucket = []
        
        for entry in filtered:
            while entry.timestamp >= current_bucket_start + interval:
                total = len(current_bucket)
                failed = sum(1 for e in current_bucket if not e.is_success)
                
                result.append({
                    'timestamp': current_bucket_start.isoformat(),
                    'total_requests': total,
                    'failed_requests': failed,
                    'error_rate': failed / total if total > 0 else 0,
                })
                
                current_bucket_start += interval
                current_bucket = []
            
            current_bucket.append(entry)
        
        # Handle last bucket
        if current_bucket:
            total = len(current_bucket)
            failed = sum(1 for e in current_bucket if not e.is_success)
            
            result.append({
                'timestamp': current_bucket_start.isoformat(),
                'total_requests': total,
                'failed_requests': failed,
                'error_rate': failed / total if total > 0 else 0,
            })
        
        return result
    
    def compute_error_breakdown_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Compute error stats for each model"""
        models = set(e.model_name for e in self.entries)
        return {model: self.compute_error_stats(model_name=model) for model in models}
    
    def find_error_patterns(self,
                            min_occurrences: int = 3,
                            model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find common error message patterns.
        
        Args:
            min_occurrences: Minimum occurrences to report
            model_name: Optional model filter
            
        Returns:
            List of error patterns with counts
        """
        failed = self.get_failed_entries(model_name=model_name)
        
        # Count error messages
        error_counts = Counter(
            e.error_message.strip() if e.error_message else 'Unknown error'
            for e in failed
        )
        
        # Filter by minimum occurrences
        patterns = [
            {
                'message': msg,
                'count': count,
                'category': self.categorize_error(msg),
            }
            for msg, count in error_counts.most_common()
            if count >= min_occurrences
        ]
        
        return patterns
    
    def find_error_bursts(self,
                          window: timedelta = timedelta(minutes=1),
                          threshold: int = 5,
                          model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find time periods with error bursts.
        
        Args:
            window: Time window for burst detection
            threshold: Minimum errors in window to consider burst
            model_name: Optional model filter
            
        Returns:
            List of error burst periods
        """
        failed = self.get_failed_entries(model_name=model_name)
        
        if not failed:
            return []
        
        failed.sort(key=lambda x: x.timestamp)
        
        bursts = []
        i = 0
        
        while i < len(failed):
            window_end = failed[i].timestamp + window
            window_errors = [failed[i]]
            
            j = i + 1
            while j < len(failed) and failed[j].timestamp < window_end:
                window_errors.append(failed[j])
                j += 1
            
            if len(window_errors) >= threshold:
                # Categorize errors in burst
                categories = defaultdict(int)
                for e in window_errors:
                    cat = self.categorize_error(e.error_message or '')
                    categories[cat or 'uncategorized'] += 1
                
                bursts.append({
                    'start_time': window_errors[0].timestamp.isoformat(),
                    'end_time': window_errors[-1].timestamp.isoformat(),
                    'error_count': len(window_errors),
                    'duration_seconds': (window_errors[-1].timestamp - window_errors[0].timestamp).total_seconds(),
                    'categories': dict(categories),
                    'sample_errors': [e.error_message for e in window_errors[:3]],
                })
                
                # Skip past this burst
                i = j
            else:
                i += 1
        
        return bursts
    
    def get_critical_errors(self,
                            model_name: Optional[str] = None) -> List[InferenceLogEntry]:
        """
        Get errors with critical or high severity.
        
        Args:
            model_name: Optional model filter
            
        Returns:
            List of critical/high severity errors
        """
        critical_categories = {
            cat.name for cat in self.categories 
            if cat.severity in ('critical', 'high')
        }
        
        failed = self.get_failed_entries(model_name=model_name)
        
        return [e for e in failed 
                if self.categorize_error(e.error_message or '') in critical_categories]
