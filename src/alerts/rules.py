"""
Alert Rules Module

Provides configurable alerting rules for inference monitoring.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
import json
import logging

from ..parsers.base import InferenceLogEntry
from ..analyzers.latency import LatencyAnalyzer
from ..analyzers.throughput import ThroughputAnalyzer
from ..analyzers.errors import ErrorAnalyzer
from ..analyzers.anomaly import AnomalyDetector, Anomaly


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


@dataclass
class Alert:
    """Alert instance"""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold: float
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'model_name': self.model_name,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged,
        }
    
    def acknowledge(self) -> None:
        """Mark alert as acknowledged"""
        self.acknowledged = True


class AlertRule(ABC):
    """Base class for alert rules"""
    
    def __init__(self, 
                 name: str,
                 severity: AlertSeverity = AlertSeverity.WARNING,
                 cooldown: timedelta = timedelta(minutes=5),
                 model_filter: Optional[str] = None):
        """
        Initialize alert rule.
        
        Args:
            name: Rule name
            severity: Alert severity
            cooldown: Minimum time between alerts
            model_filter: Optional model name filter
        """
        self.name = name
        self.severity = severity
        self.cooldown = cooldown
        self.model_filter = model_filter
        self.last_alert_time: Optional[datetime] = None
    
    @abstractmethod
    def evaluate(self, entries: List[InferenceLogEntry]) -> List[Alert]:
        """
        Evaluate the rule against entries.
        
        Args:
            entries: List of log entries to evaluate
            
        Returns:
            List of triggered alerts
        """
        pass
    
    def _should_alert(self, current_time: datetime) -> bool:
        """Check if cooldown has passed"""
        if self.last_alert_time is None:
            return True
        return current_time - self.last_alert_time >= self.cooldown
    
    def _create_alert(self,
                      message: str,
                      metric_name: str,
                      metric_value: float,
                      threshold: float,
                      model_name: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create an alert instance"""
        timestamp = datetime.now()
        self.last_alert_time = timestamp
        
        return Alert(
            rule_name=self.name,
            severity=self.severity,
            message=message,
            timestamp=timestamp,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            model_name=model_name,
            metadata=metadata or {},
        )


class LatencyThresholdRule(AlertRule):
    """Alert when latency exceeds threshold"""
    
    def __init__(self,
                 name: str = "latency_threshold",
                 p99_threshold_ms: float = 100.0,
                 p95_threshold_ms: Optional[float] = None,
                 mean_threshold_ms: Optional[float] = None,
                 severity: AlertSeverity = AlertSeverity.WARNING,
                 **kwargs):
        super().__init__(name=name, severity=severity, **kwargs)
        self.p99_threshold_ms = p99_threshold_ms
        self.p95_threshold_ms = p95_threshold_ms
        self.mean_threshold_ms = mean_threshold_ms
    
    def evaluate(self, entries: List[InferenceLogEntry]) -> List[Alert]:
        alerts = []
        
        if not entries or not self._should_alert(datetime.now()):
            return alerts
        
        analyzer = LatencyAnalyzer(entries)
        stats = analyzer.compute_stats(model_name=self.model_filter)
        
        if stats.count == 0:
            return alerts
        
        # Check P99
        if stats.p99 > self.p99_threshold_ms:
            alerts.append(self._create_alert(
                message=f"P99 latency {stats.p99:.1f}ms exceeds threshold {self.p99_threshold_ms}ms",
                metric_name="latency_p99",
                metric_value=stats.p99,
                threshold=self.p99_threshold_ms,
                model_name=self.model_filter,
                metadata={'p50': stats.p50, 'p95': stats.p95, 'count': stats.count}
            ))
        
        # Check P95
        if self.p95_threshold_ms and stats.p95 > self.p95_threshold_ms:
            alerts.append(self._create_alert(
                message=f"P95 latency {stats.p95:.1f}ms exceeds threshold {self.p95_threshold_ms}ms",
                metric_name="latency_p95",
                metric_value=stats.p95,
                threshold=self.p95_threshold_ms,
                model_name=self.model_filter,
            ))
        
        # Check mean
        if self.mean_threshold_ms and stats.mean > self.mean_threshold_ms:
            alerts.append(self._create_alert(
                message=f"Mean latency {stats.mean:.1f}ms exceeds threshold {self.mean_threshold_ms}ms",
                metric_name="latency_mean",
                metric_value=stats.mean,
                threshold=self.mean_threshold_ms,
                model_name=self.model_filter,
            ))
        
        return alerts


class ErrorRateRule(AlertRule):
    """Alert when error rate exceeds threshold"""
    
    def __init__(self,
                 name: str = "error_rate",
                 threshold_percent: float = 1.0,
                 min_requests: int = 100,
                 severity: AlertSeverity = AlertSeverity.ERROR,
                 **kwargs):
        super().__init__(name=name, severity=severity, **kwargs)
        self.threshold_percent = threshold_percent
        self.min_requests = min_requests
    
    def evaluate(self, entries: List[InferenceLogEntry]) -> List[Alert]:
        alerts = []
        
        if not entries or not self._should_alert(datetime.now()):
            return alerts
        
        # Filter by model if specified
        filtered = [e for e in entries 
                    if self.model_filter is None or e.model_name == self.model_filter]
        
        if len(filtered) < self.min_requests:
            return alerts
        
        failed = sum(1 for e in filtered if not e.is_success)
        error_rate = (failed / len(filtered)) * 100
        
        if error_rate > self.threshold_percent:
            alerts.append(self._create_alert(
                message=f"Error rate {error_rate:.2f}% exceeds threshold {self.threshold_percent}%",
                metric_name="error_rate",
                metric_value=error_rate,
                threshold=self.threshold_percent,
                model_name=self.model_filter,
                metadata={
                    'total_requests': len(filtered),
                    'failed_requests': failed,
                }
            ))
        
        return alerts


class ThroughputDropRule(AlertRule):
    """Alert when throughput drops significantly"""
    
    def __init__(self,
                 name: str = "throughput_drop",
                 drop_percent: float = 50.0,
                 baseline_window: timedelta = timedelta(hours=1),
                 evaluation_window: timedelta = timedelta(minutes=5),
                 severity: AlertSeverity = AlertSeverity.WARNING,
                 **kwargs):
        super().__init__(name=name, severity=severity, **kwargs)
        self.drop_percent = drop_percent
        self.baseline_window = baseline_window
        self.evaluation_window = evaluation_window
        self._baseline_rps: Optional[float] = None
    
    def evaluate(self, entries: List[InferenceLogEntry]) -> List[Alert]:
        alerts = []
        
        if not entries or not self._should_alert(datetime.now()):
            return alerts
        
        # Filter by model if specified
        filtered = [e for e in entries 
                    if self.model_filter is None or e.model_name == self.model_filter]
        
        if not filtered:
            return alerts
        
        now = datetime.now()
        
        # Calculate baseline
        baseline_start = now - self.baseline_window
        baseline_entries = [e for e in filtered if e.timestamp >= baseline_start]
        
        if not baseline_entries:
            return alerts
        
        baseline_duration = self.baseline_window.total_seconds()
        baseline_rps = len(baseline_entries) / baseline_duration
        
        # Calculate current
        eval_start = now - self.evaluation_window
        eval_entries = [e for e in filtered if e.timestamp >= eval_start]
        
        if not eval_entries:
            return alerts
        
        eval_duration = self.evaluation_window.total_seconds()
        current_rps = len(eval_entries) / eval_duration
        
        # Check for drop
        if baseline_rps > 0:
            drop_percent = ((baseline_rps - current_rps) / baseline_rps) * 100
            
            if drop_percent > self.drop_percent:
                alerts.append(self._create_alert(
                    message=f"Throughput dropped {drop_percent:.1f}% from {baseline_rps:.2f} to {current_rps:.2f} RPS",
                    metric_name="throughput_drop",
                    metric_value=current_rps,
                    threshold=baseline_rps * (1 - self.drop_percent / 100),
                    model_name=self.model_filter,
                    metadata={
                        'baseline_rps': baseline_rps,
                        'drop_percent': drop_percent,
                    }
                ))
        
        return alerts


class AnomalyCountRule(AlertRule):
    """Alert when too many anomalies are detected"""
    
    def __init__(self,
                 name: str = "anomaly_count",
                 threshold_count: int = 10,
                 evaluation_window: timedelta = timedelta(minutes=15),
                 severity: AlertSeverity = AlertSeverity.WARNING,
                 **kwargs):
        super().__init__(name=name, severity=severity, **kwargs)
        self.threshold_count = threshold_count
        self.evaluation_window = evaluation_window
    
    def evaluate(self, entries: List[InferenceLogEntry]) -> List[Alert]:
        alerts = []
        
        if not entries or not self._should_alert(datetime.now()):
            return alerts
        
        # Filter recent entries
        now = datetime.now()
        window_start = now - self.evaluation_window
        recent = [e for e in entries if e.timestamp >= window_start]
        
        if not recent:
            return alerts
        
        # Detect anomalies
        detector = AnomalyDetector(recent)
        anomalies = detector.detect_all_anomalies(model_name=self.model_filter)
        
        if len(anomalies) > self.threshold_count:
            # Group by type
            by_type = {}
            for a in anomalies:
                type_name = a.anomaly_type.value
                by_type[type_name] = by_type.get(type_name, 0) + 1
            
            alerts.append(self._create_alert(
                message=f"Detected {len(anomalies)} anomalies in {self.evaluation_window.total_seconds()/60:.0f} minutes",
                metric_name="anomaly_count",
                metric_value=len(anomalies),
                threshold=self.threshold_count,
                model_name=self.model_filter,
                metadata={
                    'by_type': by_type,
                    'window_minutes': self.evaluation_window.total_seconds() / 60,
                }
            ))
        
        return alerts


class AlertManager:
    """Manages alert rules and notifications"""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.alert_history: List[Alert] = []
        self.handlers: List[Callable[[Alert], None]] = []
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule"""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name"""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                logger.info(f"Removed alert rule: {name}")
                return True
        return False
    
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler callback"""
        self.handlers.append(handler)
    
    def evaluate(self, entries: List[InferenceLogEntry]) -> List[Alert]:
        """
        Evaluate all rules against entries.
        
        Args:
            entries: List of log entries to evaluate
            
        Returns:
            List of triggered alerts
        """
        all_alerts = []
        
        for rule in self.rules:
            try:
                alerts = rule.evaluate(entries)
                for alert in alerts:
                    self.alert_history.append(alert)
                    all_alerts.append(alert)
                    
                    # Call handlers
                    for handler in self.handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logger.error(f"Alert handler error: {e}")
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        return all_alerts
    
    def get_active_alerts(self, 
                          since: Optional[datetime] = None,
                          severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active (unacknowledged) alerts"""
        alerts = [a for a in self.alert_history if not a.acknowledged]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def acknowledge_alert(self, 
                          rule_name: Optional[str] = None,
                          before: Optional[datetime] = None) -> int:
        """
        Acknowledge alerts.
        
        Args:
            rule_name: Optional rule name filter
            before: Acknowledge alerts before this time
            
        Returns:
            Number of alerts acknowledged
        """
        count = 0
        for alert in self.alert_history:
            if alert.acknowledged:
                continue
            if rule_name and alert.rule_name != rule_name:
                continue
            if before and alert.timestamp >= before:
                continue
            
            alert.acknowledge()
            count += 1
        
        return count
    
    def export_alerts(self, filepath: str) -> None:
        """Export alert history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump([a.to_dict() for a in self.alert_history], f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of alert history"""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'active_alerts': 0,
                'by_severity': {},
                'by_rule': {},
            }
        
        active = [a for a in self.alert_history if not a.acknowledged]
        
        by_severity = {}
        for sev in AlertSeverity:
            by_severity[sev.value] = len([a for a in active if a.severity == sev])
        
        by_rule = {}
        for rule in self.rules:
            by_rule[rule.name] = len([a for a in active if a.rule_name == rule.name])
        
        return {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(active),
            'by_severity': by_severity,
            'by_rule': by_rule,
        }


# Convenience functions for common alert handlers

def console_handler(alert: Alert) -> None:
    """Print alert to console"""
    severity_colors = {
        AlertSeverity.INFO: '\033[94m',
        AlertSeverity.WARNING: '\033[93m', 
        AlertSeverity.ERROR: '\033[91m',
        AlertSeverity.CRITICAL: '\033[91m\033[1m',
    }
    reset = '\033[0m'
    color = severity_colors.get(alert.severity, '')
    
    print(f"{color}[{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}{reset}")


def file_handler(filepath: str) -> Callable[[Alert], None]:
    """Create a handler that logs alerts to a file"""
    def handler(alert: Alert) -> None:
        with open(filepath, 'a') as f:
            f.write(json.dumps(alert.to_dict()) + '\n')
    return handler


def webhook_handler(url: str) -> Callable[[Alert], None]:
    """Create a handler that sends alerts to a webhook"""
    import urllib.request
    
    def handler(alert: Alert) -> None:
        try:
            data = json.dumps(alert.to_dict()).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.error(f"Webhook handler error: {e}")
    
    return handler
