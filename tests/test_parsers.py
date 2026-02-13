"""
Test Suite for Inference Log Analytics

Comprehensive tests for parsers, analyzers, and alerting.
"""

import pytest
import json
from datetime import datetime, timedelta
from typing import List

# Import modules to test
from src.parsers.base import BaseParser, InferenceLogEntry
from src.parsers.onnx_runtime import ONNXRuntimeParser
from src.parsers.tensorflow_serving import TensorFlowServingParser
from src.parsers.triton import TritonParser
from src.parsers.custom import CustomParser
from src.parsers import auto_detect_parser, get_parser

from src.analyzers.latency import LatencyAnalyzer, LatencyStats
from src.analyzers.throughput import ThroughputAnalyzer, ThroughputStats
from src.analyzers.errors import ErrorAnalyzer, ErrorCategory
from src.analyzers.anomaly import AnomalyDetector, Anomaly, AnomalyType

from src.alerts.rules import (
    AlertManager, AlertRule, Alert, AlertSeverity,
    LatencyThresholdRule, ErrorRateRule, ThroughputDropRule
)


# ============ Test Fixtures ============

@pytest.fixture
def sample_entries() -> List[InferenceLogEntry]:
    """Generate sample inference log entries"""
    base_time = datetime.now()
    entries = []
    
    for i in range(100):
        # Normal requests
        entries.append(InferenceLogEntry(
            timestamp=base_time + timedelta(seconds=i),
            request_id=f"req_{i:04d}",
            model_name="bert-base",
            latency_ms=20 + (i % 10),
            status="success",
            is_success=True,
            batch_size=1,
        ))
    
    # Add some slow requests
    for i in range(5):
        entries.append(InferenceLogEntry(
            timestamp=base_time + timedelta(seconds=100 + i),
            request_id=f"slow_{i:04d}",
            model_name="bert-base",
            latency_ms=200 + i * 50,
            status="success",
            is_success=True,
        ))
    
    # Add some failures
    for i in range(10):
        entries.append(InferenceLogEntry(
            timestamp=base_time + timedelta(seconds=110 + i),
            request_id=f"fail_{i:04d}",
            model_name="bert-base",
            latency_ms=0,
            status="error",
            is_success=False,
            error_message="Timeout error" if i < 5 else "GPU memory error",
        ))
    
    return entries


@pytest.fixture
def sample_json_log() -> str:
    """Sample JSON log line"""
    return json.dumps({
        "timestamp": "2024-01-15T10:30:00.000Z",
        "request_id": "req_001",
        "model_name": "gpt-2",
        "latency_ms": 45.5,
        "status": "success",
        "batch_size": 4
    })


@pytest.fixture
def sample_onnx_logs() -> List[str]:
    """Sample ONNX Runtime logs"""
    return [
        '{"timestamp": "2024-01-15 10:30:00", "request_id": "onnx_001", "model_name": "resnet50", "duration_ms": 25.3, "status": "success"}',
        '{"timestamp": "2024-01-15 10:30:01", "request_id": "onnx_002", "model_name": "resnet50", "duration_ms": 28.1, "status": "success"}',
        '{"timestamp": "2024-01-15 10:30:02", "request_id": "onnx_003", "model_name": "resnet50", "duration_ms": 150.0, "status": "error", "error": "OOM"}',
    ]


# ============ Parser Tests ============

class TestInferenceLogEntry:
    """Tests for InferenceLogEntry dataclass"""
    
    def test_creation(self):
        entry = InferenceLogEntry(
            timestamp=datetime.now(),
            request_id="test_001",
            model_name="test_model",
            latency_ms=50.0,
            status="success",
            is_success=True,
        )
        assert entry.request_id == "test_001"
        assert entry.latency_ms == 50.0
        assert entry.is_success is True
    
    def test_to_dict(self):
        entry = InferenceLogEntry(
            timestamp=datetime.now(),
            request_id="test_001",
            model_name="test_model",
            latency_ms=50.0,
            status="success",
            is_success=True,
        )
        d = entry.to_dict()
        assert d['request_id'] == "test_001"
        assert d['latency_ms'] == 50.0


class TestONNXRuntimeParser:
    """Tests for ONNX Runtime parser"""
    
    def test_parse_json_line(self, sample_onnx_logs):
        parser = ONNXRuntimeParser()
        entry = parser.parse_line(sample_onnx_logs[0])
        
        assert entry is not None
        assert entry.model_name == "resnet50"
        assert entry.latency_ms == 25.3
        assert entry.is_success is True
    
    def test_parse_error_log(self, sample_onnx_logs):
        parser = ONNXRuntimeParser()
        entry = parser.parse_line(sample_onnx_logs[2])
        
        assert entry is not None
        assert entry.is_success is False
        assert "OOM" in entry.error_message
    
    def test_parse_file(self, sample_onnx_logs, tmp_path):
        log_file = tmp_path / "onnx.log"
        log_file.write_text("\n".join(sample_onnx_logs))
        
        parser = ONNXRuntimeParser()
        entries = parser.parse_file(str(log_file))
        
        assert len(entries) == 3
        assert entries[0].model_name == "resnet50"


class TestCustomParser:
    """Tests for custom parser"""
    
    def test_json_parsing(self, sample_json_log):
        parser = CustomParser()
        entry = parser.parse_line(sample_json_log)
        
        assert entry is not None
        assert entry.model_name == "gpt-2"
        assert entry.latency_ms == 45.5
    
    def test_custom_field_mapping(self):
        log_line = json.dumps({
            "ts": "2024-01-15T10:30:00Z",
            "id": "custom_001",
            "model": "custom_model",
            "duration": 30.0,
            "result": "ok"
        })
        
        parser = CustomParser(field_mapping={
            'timestamp': 'ts',
            'request_id': 'id',
            'model_name': 'model',
            'latency_ms': 'duration',
            'status': 'result',
        })
        
        entry = parser.parse_line(log_line)
        assert entry is not None
        assert entry.request_id == "custom_001"
        assert entry.model_name == "custom_model"


class TestParserRegistry:
    """Tests for parser registry functions"""
    
    def test_get_parser(self):
        parser = get_parser("onnx_runtime")
        assert isinstance(parser, ONNXRuntimeParser)
        
        parser = get_parser("tensorflow_serving")
        assert isinstance(parser, TensorFlowServingParser)
    
    def test_auto_detect(self, sample_onnx_logs, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("\n".join(sample_onnx_logs))
        
        parser = auto_detect_parser(str(log_file))
        assert parser is not None


# ============ Analyzer Tests ============

class TestLatencyAnalyzer:
    """Tests for latency analyzer"""
    
    def test_compute_stats(self, sample_entries):
        analyzer = LatencyAnalyzer(sample_entries)
        stats = analyzer.compute_stats()
        
        assert stats.count > 0
        assert stats.mean > 0
        assert stats.p50 > 0
        assert stats.p99 >= stats.p50
    
    def test_compute_stats_by_model(self, sample_entries):
        analyzer = LatencyAnalyzer(sample_entries)
        by_model = analyzer.compute_stats_by_model()
        
        assert "bert-base" in by_model
        assert by_model["bert-base"].count > 0
    
    def test_find_slow_requests(self, sample_entries):
        analyzer = LatencyAnalyzer(sample_entries)
        slow = analyzer.find_slow_requests(threshold_percentile=99)
        
        # Should find the intentionally slow requests
        assert len(slow) > 0
    
    def test_time_series(self, sample_entries):
        analyzer = LatencyAnalyzer(sample_entries)
        ts = analyzer.compute_time_series(interval=timedelta(seconds=30))
        
        assert len(ts) > 0
        assert 'timestamp' in ts[0]
        assert 'mean' in ts[0]


class TestThroughputAnalyzer:
    """Tests for throughput analyzer"""
    
    def test_compute_stats(self, sample_entries):
        analyzer = ThroughputAnalyzer(sample_entries)
        stats = analyzer.compute_stats()
        
        assert stats.total_requests == len(sample_entries)
        assert stats.successful_requests > 0
        assert stats.requests_per_second > 0
    
    def test_hourly_breakdown(self, sample_entries):
        analyzer = ThroughputAnalyzer(sample_entries)
        breakdown = analyzer.compute_hourly_breakdown()
        
        assert len(breakdown) == 24
        # At least one hour should have data
        assert any(breakdown[h]['total_requests'] > 0 for h in range(24))


class TestErrorAnalyzer:
    """Tests for error analyzer"""
    
    def test_compute_error_stats(self, sample_entries):
        analyzer = ErrorAnalyzer(sample_entries)
        stats = analyzer.compute_error_stats()
        
        assert stats['failed_requests'] == 10  # We added 10 failures
        assert stats['error_rate'] > 0
    
    def test_categorize_error(self, sample_entries):
        analyzer = ErrorAnalyzer(sample_entries)
        
        category = analyzer.categorize_error("Timeout error occurred")
        assert category == "timeout"
        
        category = analyzer.categorize_error("CUDA out of memory")
        assert category == "memory"
    
    def test_error_breakdown(self, sample_entries):
        analyzer = ErrorAnalyzer(sample_entries)
        breakdown = analyzer.compute_error_breakdown_by_category()
        
        # Should have categorized errors
        assert len(breakdown) > 0


class TestAnomalyDetector:
    """Tests for anomaly detector"""
    
    def test_detect_latency_anomalies(self, sample_entries):
        detector = AnomalyDetector(sample_entries)
        anomalies = detector.detect_latency_anomalies()
        
        # Should detect the slow requests as anomalies
        assert any(a.anomaly_type == AnomalyType.LATENCY_SPIKE for a in anomalies)
    
    def test_get_anomaly_summary(self, sample_entries):
        detector = AnomalyDetector(sample_entries)
        summary = detector.get_anomaly_summary()
        
        assert 'total_anomalies' in summary
        assert 'by_type' in summary


# ============ Alert Tests ============

class TestLatencyThresholdRule:
    """Tests for latency threshold alert rule"""
    
    def test_trigger_alert(self, sample_entries):
        # Set threshold low enough to trigger
        rule = LatencyThresholdRule(
            p99_threshold_ms=50.0,
            severity=AlertSeverity.WARNING
        )
        
        alerts = rule.evaluate(sample_entries)
        assert len(alerts) > 0
        assert alerts[0].severity == AlertSeverity.WARNING
    
    def test_no_alert_within_threshold(self, sample_entries):
        # Set threshold high enough to not trigger
        rule = LatencyThresholdRule(
            p99_threshold_ms=1000.0,
            severity=AlertSeverity.WARNING
        )
        
        # First evaluation triggers normally but..
        # Need entries with lower p99
        low_latency_entries = [
            InferenceLogEntry(
                timestamp=datetime.now() + timedelta(seconds=i),
                request_id=f"low_{i}",
                model_name="test",
                latency_ms=10 + i,
                status="success",
                is_success=True,
            )
            for i in range(100)
        ]
        
        alerts = rule.evaluate(low_latency_entries)
        assert len(alerts) == 0


class TestErrorRateRule:
    """Tests for error rate alert rule"""
    
    def test_trigger_alert(self, sample_entries):
        rule = ErrorRateRule(
            threshold_percent=5.0,  # sample has ~9% error rate
            min_requests=50,
            severity=AlertSeverity.ERROR
        )
        
        alerts = rule.evaluate(sample_entries)
        assert len(alerts) > 0
        assert alerts[0].metric_name == "error_rate"


class TestAlertManager:
    """Tests for alert manager"""
    
    def test_add_rule(self):
        manager = AlertManager()
        rule = LatencyThresholdRule(p99_threshold_ms=100.0)
        
        manager.add_rule(rule)
        assert len(manager.rules) == 1
    
    def test_evaluate_rules(self, sample_entries):
        manager = AlertManager()
        manager.add_rule(LatencyThresholdRule(p99_threshold_ms=50.0))
        manager.add_rule(ErrorRateRule(threshold_percent=5.0, min_requests=50))
        
        alerts = manager.evaluate(sample_entries)
        assert len(alerts) > 0
        assert len(manager.alert_history) > 0
    
    def test_acknowledge_alerts(self, sample_entries):
        manager = AlertManager()
        manager.add_rule(LatencyThresholdRule(p99_threshold_ms=50.0))
        
        manager.evaluate(sample_entries)
        active_before = len(manager.get_active_alerts())
        
        manager.acknowledge_alert(rule_name="latency_threshold")
        active_after = len(manager.get_active_alerts())
        
        assert active_after < active_before


# ============ Integration Tests ============

class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_full_pipeline(self, sample_entries, tmp_path):
        """Test complete analysis pipeline"""
        from src.visualizers.dashboard import Dashboard
        
        # Create dashboard
        dashboard = Dashboard(sample_entries)
        
        # Get summary
        summary = dashboard.get_summary()
        assert summary['request_counts']['total'] == len(sample_entries)
        
        # Generate report
        report_path = dashboard.generate_report(
            output_dir=str(tmp_path),
            format='json',
            include_plots=False
        )
        
        assert (tmp_path / "report.json").exists()


# ============ Run Tests ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
