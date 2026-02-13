"""
Dashboard Module

Provides a comprehensive dashboard for inference analytics.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from ..parsers.base import InferenceLogEntry
from ..analyzers.latency import LatencyAnalyzer
from ..analyzers.throughput import ThroughputAnalyzer
from ..analyzers.errors import ErrorAnalyzer
from ..analyzers.anomaly import AnomalyDetector
from .plots import (
    plot_latency_histogram,
    plot_latency_time_series,
    plot_throughput_time_series,
    plot_error_rates,
    plot_anomalies,
    plot_model_comparison,
)


class Dashboard:
    """
    Comprehensive dashboard for inference log analytics.
    
    Aggregates multiple analyzers and provides unified reporting
    and visualization capabilities.
    """
    
    def __init__(self, entries: Optional[List[InferenceLogEntry]] = None):
        """
        Initialize dashboard with optional entries.
        
        Args:
            entries: List of inference log entries
        """
        self.entries = entries or []
        
        # Initialize analyzers
        self._latency_analyzer = None
        self._throughput_analyzer = None
        self._error_analyzer = None
        self._anomaly_detector = None
        
        if entries:
            self._init_analyzers()
    
    def _init_analyzers(self) -> None:
        """Initialize all analyzers with current entries"""
        self._latency_analyzer = LatencyAnalyzer(self.entries)
        self._throughput_analyzer = ThroughputAnalyzer(self.entries)
        self._error_analyzer = ErrorAnalyzer(self.entries)
        self._anomaly_detector = AnomalyDetector(self.entries)
    
    def add_entries(self, entries: List[InferenceLogEntry]) -> None:
        """
        Add entries to the dashboard.
        
        Args:
            entries: List of entries to add
        """
        self.entries.extend(entries)
        self._init_analyzers()
    
    @property
    def latency_analyzer(self) -> LatencyAnalyzer:
        """Get latency analyzer"""
        if self._latency_analyzer is None:
            self._latency_analyzer = LatencyAnalyzer(self.entries)
        return self._latency_analyzer
    
    @property
    def throughput_analyzer(self) -> ThroughputAnalyzer:
        """Get throughput analyzer"""
        if self._throughput_analyzer is None:
            self._throughput_analyzer = ThroughputAnalyzer(self.entries)
        return self._throughput_analyzer
    
    @property
    def error_analyzer(self) -> ErrorAnalyzer:
        """Get error analyzer"""
        if self._error_analyzer is None:
            self._error_analyzer = ErrorAnalyzer(self.entries)
        return self._error_analyzer
    
    @property
    def anomaly_detector(self) -> AnomalyDetector:
        """Get anomaly detector"""
        if self._anomaly_detector is None:
            self._anomaly_detector = AnomalyDetector(self.entries)
        return self._anomaly_detector
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall summary of inference logs.
        
        Returns:
            Dict with summary statistics
        """
        if not self.entries:
            return {'error': 'No entries to analyze'}
        
        # Time range
        timestamps = [e.timestamp for e in self.entries]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        # Basic counts
        total = len(self.entries)
        successful = sum(1 for e in self.entries if e.is_success)
        failed = total - successful
        
        # Latency stats
        latency_stats = self.latency_analyzer.compute_stats()
        
        # Throughput stats
        throughput_stats = self.throughput_analyzer.compute_stats()
        
        # Error stats
        error_stats = self.error_analyzer.compute_error_stats()
        
        # Anomaly summary
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        
        # Model breakdown
        models = list(set(e.model_name for e in self.entries))
        
        return {
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600,
            },
            'request_counts': {
                'total': total,
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total if total > 0 else 0,
            },
            'latency': latency_stats.to_dict(),
            'throughput': throughput_stats.to_dict(),
            'errors': error_stats,
            'anomalies': anomaly_summary,
            'models': models,
            'model_count': len(models),
        }
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """
        Get summary for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict with model-specific statistics
        """
        model_entries = [e for e in self.entries if e.model_name == model_name]
        
        if not model_entries:
            return {'error': f'No entries for model: {model_name}'}
        
        return {
            'model_name': model_name,
            'latency': self.latency_analyzer.compute_stats(model_name=model_name).to_dict(),
            'throughput': self.throughput_analyzer.compute_stats(model_name=model_name).to_dict(),
            'errors': self.error_analyzer.compute_error_stats(model_name=model_name),
        }
    
    def generate_report(self,
                        output_dir: str,
                        format: str = 'html',
                        include_plots: bool = True) -> str:
        """
        Generate a comprehensive report.
        
        Args:
            output_dir: Directory to save report files
            format: Report format ('html', 'json', 'markdown')
            include_plots: Whether to include visualizations
            
        Returns:
            Path to the generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_summary()
        
        # Generate plots if requested
        plot_paths = []
        if include_plots:
            plots_dir = output_path / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Latency histogram
            fig = plot_latency_histogram(self.entries)
            hist_path = plots_dir / 'latency_histogram.png'
            fig.savefig(hist_path, dpi=150, bbox_inches='tight')
            plot_paths.append(str(hist_path))
            
            # Latency time series
            ts_data = self.latency_analyzer.compute_time_series()
            fig = plot_latency_time_series(ts_data)
            ts_path = plots_dir / 'latency_time_series.png'
            fig.savefig(ts_path, dpi=150, bbox_inches='tight')
            plot_paths.append(str(ts_path))
            
            # Throughput time series
            tp_data = self.throughput_analyzer.compute_time_series()
            fig = plot_throughput_time_series(tp_data)
            tp_path = plots_dir / 'throughput_time_series.png'
            fig.savefig(tp_path, dpi=150, bbox_inches='tight')
            plot_paths.append(str(tp_path))
            
            # Error rates
            err_data = self.error_analyzer.compute_error_time_series()
            fig = plot_error_rates(err_data)
            err_path = plots_dir / 'error_rates.png'
            fig.savefig(err_path, dpi=150, bbox_inches='tight')
            plot_paths.append(str(err_path))
        
        if format == 'json':
            report_path = output_path / 'report.json'
            with open(report_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        elif format == 'markdown':
            report_path = output_path / 'report.md'
            md_content = self._generate_markdown_report(summary, plot_paths)
            with open(report_path, 'w') as f:
                f.write(md_content)
        
        else:  # html
            report_path = output_path / 'report.html'
            html_content = self._generate_html_report(summary, plot_paths)
            with open(report_path, 'w') as f:
                f.write(html_content)
        
        return str(report_path)
    
    def _generate_markdown_report(self, 
                                   summary: Dict[str, Any],
                                   plot_paths: List[str]) -> str:
        """Generate markdown report content"""
        lines = [
            "# Inference Analytics Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"- **Time Range:** {summary['time_range']['start']} to {summary['time_range']['end']}",
            f"- **Duration:** {summary['time_range']['duration_hours']:.2f} hours",
            f"- **Total Requests:** {summary['request_counts']['total']:,}",
            f"- **Success Rate:** {summary['request_counts']['success_rate']*100:.2f}%",
            "",
            "## Latency Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean | {summary['latency']['mean']:.2f} ms |",
            f"| Median | {summary['latency']['median']:.2f} ms |",
            f"| P95 | {summary['latency']['p95']:.2f} ms |",
            f"| P99 | {summary['latency']['p99']:.2f} ms |",
            "",
            "## Throughput Statistics",
            "",
            f"- **Requests/Second:** {summary['throughput']['requests_per_second']:.2f}",
            f"- **Peak RPS:** {summary['throughput']['peak_rps']:.2f}",
            "",
            "## Error Analysis",
            "",
            f"- **Failed Requests:** {summary['errors']['failed_requests']}",
            f"- **Error Rate:** {summary['errors']['error_rate']*100:.2f}%",
            "",
        ]
        
        if summary['errors'].get('categories'):
            lines.append("### Error Categories")
            lines.append("")
            for cat, count in summary['errors']['categories'].items():
                lines.append(f"- **{cat}:** {count}")
            lines.append("")
        
        lines.extend([
            "## Anomalies",
            "",
            f"- **Total Anomalies:** {summary['anomalies']['total_anomalies']}",
            "",
        ])
        
        if plot_paths:
            lines.append("## Visualizations")
            lines.append("")
            for path in plot_paths:
                name = os.path.basename(path).replace('_', ' ').replace('.png', '').title()
                lines.append(f"### {name}")
                lines.append(f"![{name}]({path})")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(self,
                               summary: Dict[str, Any],
                               plot_paths: List[str]) -> str:
        """Generate HTML report content"""
        
        plots_html = ""
        for path in plot_paths:
            name = os.path.basename(path).replace('_', ' ').replace('.png', '').title()
            # Use relative path for plots
            rel_path = os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)
            plots_html += f'''
            <div class="plot">
                <h3>{name}</h3>
                <img src="{rel_path}" alt="{name}">
            </div>
            '''
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Inference Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .stat-card.warning {{ border-left-color: #ff9800; }}
        .stat-card.error {{ border-left-color: #f44336; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .plot {{ margin: 20px 0; }}
        .plot img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Inference Analytics Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overview</h2>
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value">{summary['request_counts']['total']:,}</div>
                <div class="stat-label">Total Requests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['request_counts']['success_rate']*100:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['latency']['p50']:.1f}ms</div>
                <div class="stat-label">P50 Latency</div>
            </div>
            <div class="stat-card {'warning' if summary['latency']['p99'] > 100 else ''}">
                <div class="stat-value">{summary['latency']['p99']:.1f}ms</div>
                <div class="stat-label">P99 Latency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['throughput']['requests_per_second']:.1f}</div>
                <div class="stat-label">Avg RPS</div>
            </div>
            <div class="stat-card {'error' if summary['anomalies']['total_anomalies'] > 10 else ''}">
                <div class="stat-value">{summary['anomalies']['total_anomalies']}</div>
                <div class="stat-label">Anomalies Detected</div>
            </div>
        </div>
        
        <h2>Latency Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Mean</td><td>{summary['latency']['mean']:.2f} ms</td></tr>
            <tr><td>Median</td><td>{summary['latency']['median']:.2f} ms</td></tr>
            <tr><td>Std Dev</td><td>{summary['latency']['std']:.2f} ms</td></tr>
            <tr><td>Min</td><td>{summary['latency']['min']:.2f} ms</td></tr>
            <tr><td>Max</td><td>{summary['latency']['max']:.2f} ms</td></tr>
            <tr><td>P50</td><td>{summary['latency']['p50']:.2f} ms</td></tr>
            <tr><td>P90</td><td>{summary['latency']['p90']:.2f} ms</td></tr>
            <tr><td>P95</td><td>{summary['latency']['p95']:.2f} ms</td></tr>
            <tr><td>P99</td><td>{summary['latency']['p99']:.2f} ms</td></tr>
        </table>
        
        <h2>Throughput</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Duration</td><td>{summary['throughput']['duration_seconds']:.0f} seconds</td></tr>
            <tr><td>Average RPS</td><td>{summary['throughput']['requests_per_second']:.2f}</td></tr>
            <tr><td>Peak RPS</td><td>{summary['throughput']['peak_rps']:.2f}</td></tr>
        </table>
        
        <h2>Error Analysis</h2>
        <div class="summary-grid">
            <div class="stat-card {'error' if summary['errors']['error_rate'] > 0.01 else ''}">
                <div class="stat-value">{summary['errors']['failed_requests']}</div>
                <div class="stat-label">Failed Requests</div>
            </div>
            <div class="stat-card {'error' if summary['errors']['error_rate'] > 0.01 else ''}">
                <div class="stat-value">{summary['errors']['error_rate']*100:.2f}%</div>
                <div class="stat-label">Error Rate</div>
            </div>
        </div>
        
        <h2>Visualizations</h2>
        {plots_html}
        
        <h2>Models Analyzed</h2>
        <p>{', '.join(summary['models'])}</p>
    </div>
</body>
</html>'''
        
        return html
    
    def create_live_dashboard(self, 
                               refresh_interval: int = 5,
                               port: int = 8080) -> None:
        """
        Start a live dashboard server (placeholder for web implementation).
        
        Args:
            refresh_interval: Dashboard refresh interval in seconds
            port: Port to run the server on
        """
        # This would be implemented with Flask/FastAPI in a real deployment
        print(f"Live dashboard would start on port {port}")
        print(f"Refresh interval: {refresh_interval} seconds")
        print("Note: Full web server implementation requires Flask/FastAPI")
