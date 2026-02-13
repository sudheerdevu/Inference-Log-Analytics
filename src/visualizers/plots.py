"""
Plotting Functions Module

Provides matplotlib-based visualization functions for inference analytics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..parsers.base import InferenceLogEntry
from ..analyzers.anomaly import Anomaly


def setup_figure(figsize: Tuple[int, int] = (12, 6), 
                 title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Setup a matplotlib figure with common styling"""
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_latency_histogram(entries: List[InferenceLogEntry],
                           model_name: Optional[str] = None,
                           bins: int = 50,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot latency histogram.
    
    Args:
        entries: List of inference log entries
        model_name: Optional model filter
        bins: Number of histogram bins
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    latencies = [e.latency_ms for e in entries 
                 if e.is_success and (model_name is None or e.model_name == model_name)]
    
    if not latencies:
        fig, ax = setup_figure(title="No data available")
        return fig
    
    title = title or f"Latency Distribution{f' - {model_name}' if model_name else ''}"
    fig, ax = setup_figure(title=title)
    
    ax.hist(latencies, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add percentile lines
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    ax.axvline(p50, color='green', linestyle='--', label=f'P50: {p50:.1f}ms')
    ax.axvline(p95, color='orange', linestyle='--', label=f'P95: {p95:.1f}ms')
    ax.axvline(p99, color='red', linestyle='--', label=f'P99: {p99:.1f}ms')
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Count')
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_latency_time_series(time_series: List[Dict[str, Any]],
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot latency time series.
    
    Args:
        time_series: List of time series data points
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    if not time_series:
        fig, ax = setup_figure(title="No data available")
        return fig
    
    timestamps = [datetime.fromisoformat(ts['timestamp']) for ts in time_series]
    p50 = [ts.get('p50', ts.get('mean', 0)) for ts in time_series]
    p95 = [ts.get('p95', 0) for ts in time_series]
    p99 = [ts.get('p99', 0) for ts in time_series]
    
    title = title or "Latency Over Time"
    fig, ax = setup_figure(title=title)
    
    ax.plot(timestamps, p50, label='P50', color='green', linewidth=2)
    ax.plot(timestamps, p95, label='P95', color='orange', linewidth=2)
    ax.plot(timestamps, p99, label='P99', color='red', linewidth=2)
    
    ax.fill_between(timestamps, p50, p99, alpha=0.2, color='gray')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Latency (ms)')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_throughput_time_series(time_series: List[Dict[str, Any]],
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot throughput time series.
    
    Args:
        time_series: List of time series data points
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    if not time_series:
        fig, ax = setup_figure(title="No data available")
        return fig
    
    timestamps = [datetime.fromisoformat(ts['timestamp']) for ts in time_series]
    rps = [ts.get('rps', ts.get('total_requests', 0)) for ts in time_series]
    success_rate = [ts.get('success_rate', 1.0) * 100 for ts in time_series]
    
    title = title or "Throughput Over Time"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # RPS plot
    ax1.plot(timestamps, rps, label='Requests/sec', color='steelblue', linewidth=2)
    ax1.fill_between(timestamps, 0, rps, alpha=0.3, color='steelblue')
    ax1.set_ylabel('Requests per Second')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Success rate plot
    ax2.plot(timestamps, success_rate, label='Success Rate', color='green', linewidth=2)
    ax2.axhline(99, color='orange', linestyle='--', alpha=0.7, label='99% threshold')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.set_xlabel('Time')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_rates(error_time_series: List[Dict[str, Any]],
                     title: Optional[str] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot error rates over time.
    
    Args:
        error_time_series: List of error rate data points
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    if not error_time_series:
        fig, ax = setup_figure(title="No data available")
        return fig
    
    timestamps = [datetime.fromisoformat(ts['timestamp']) for ts in error_time_series]
    error_rates = [ts.get('error_rate', 0) * 100 for ts in error_time_series]
    failed_counts = [ts.get('failed_requests', 0) for ts in error_time_series]
    
    title = title or "Error Rates Over Time"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Error rate plot
    ax1.plot(timestamps, error_rates, label='Error Rate', color='red', linewidth=2)
    ax1.fill_between(timestamps, 0, error_rates, alpha=0.3, color='red')
    ax1.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='1% threshold')
    ax1.set_ylabel('Error Rate (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Failed count plot
    ax2.bar(timestamps, failed_counts, color='coral', alpha=0.7, label='Failed Requests')
    ax2.set_ylabel('Failed Request Count')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.set_xlabel('Time')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_anomalies(entries: List[InferenceLogEntry],
                   anomalies: List[Anomaly],
                   metric: str = 'latency',
                   title: Optional[str] = None,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time series with anomalies highlighted.
    
    Args:
        entries: List of inference log entries
        anomalies: List of detected anomalies
        metric: Which metric to plot ('latency', 'throughput')
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    if not entries:
        fig, ax = setup_figure(title="No data available")
        return fig
    
    title = title or f"Anomaly Detection - {metric.capitalize()}"
    fig, ax = setup_figure(title=title, figsize=(14, 6))
    
    if metric == 'latency':
        successful = [(e.timestamp, e.latency_ms) for e in entries if e.is_success]
        successful.sort(key=lambda x: x[0])
        
        times = [t for t, _ in successful]
        values = [v for _, v in successful]
        
        ax.plot(times, values, color='steelblue', alpha=0.6, linewidth=1, label='Latency')
        ax.set_ylabel('Latency (ms)')
    
    # Highlight anomalies
    anomaly_times = [a.timestamp for a in anomalies]
    anomaly_values = [a.metric_value for a in anomalies]
    
    if anomaly_times:
        # Color by severity
        colors = ['yellow' if a.severity < 0.5 else 'orange' if a.severity < 0.75 else 'red' 
                  for a in anomalies]
        ax.scatter(anomaly_times, anomaly_values, c=colors, s=100, 
                   marker='o', edgecolors='black', label='Anomalies', zorder=5)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.set_xlabel('Time')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(stats_by_model: Dict[str, Dict[str, Any]],
                           metric: str = 'latency',
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison across models.
    
    Args:
        stats_by_model: Dict mapping model name to stats
        metric: Which metric to compare
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    if not stats_by_model:
        fig, ax = setup_figure(title="No data available")
        return fig
    
    models = list(stats_by_model.keys())
    x = np.arange(len(models))
    
    title = title or f"Model Comparison - {metric.capitalize()}"
    fig, ax = setup_figure(title=title, figsize=(10, 6))
    
    if metric == 'latency':
        means = [stats_by_model[m].get('mean', 0) for m in models]
        p95s = [stats_by_model[m].get('p95', 0) for m in models]
        p99s = [stats_by_model[m].get('p99', 0) for m in models]
        
        width = 0.25
        ax.bar(x - width, means, width, label='Mean', color='steelblue')
        ax.bar(x, p95s, width, label='P95', color='orange')
        ax.bar(x + width, p99s, width, label='P99', color='red')
        
        ax.set_ylabel('Latency (ms)')
    elif metric == 'throughput':
        rps = [stats_by_model[m].get('requests_per_second', 0) for m in models]
        ax.bar(x, rps, color='steelblue')
        ax.set_ylabel('Requests per Second')
    elif metric == 'error_rate':
        rates = [stats_by_model[m].get('error_rate', 0) * 100 for m in models]
        ax.bar(x, rates, color='coral')
        ax.set_ylabel('Error Rate (%)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_xlabel('Model')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_error_category_breakdown(error_breakdown: Dict[str, Dict[str, Any]],
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot error breakdown by category.
    
    Args:
        error_breakdown: Dict mapping category to stats
        title: Optional custom title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    if not error_breakdown:
        fig, ax = setup_figure(title="No errors")
        return fig
    
    categories = list(error_breakdown.keys())
    counts = [error_breakdown[c].get('count', 0) for c in categories]
    
    # Color by severity
    severity_colors = {
        'low': 'green',
        'medium': 'yellow',
        'high': 'orange',
        'critical': 'red',
    }
    colors = [severity_colors.get(error_breakdown[c].get('severity', 'medium'), 'gray') 
              for c in categories]
    
    title = title or "Error Breakdown by Category"
    fig, ax = setup_figure(title=title)
    
    ax.barh(categories, counts, color=colors, edgecolor='black')
    ax.set_xlabel('Count')
    ax.set_ylabel('Error Category')
    
    # Add count labels
    for i, (count, cat) in enumerate(zip(counts, categories)):
        ax.text(count + max(counts) * 0.01, i, str(count), va='center')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
