# Inference Log Analytics Engine

> Production-grade toolkit for analyzing AI inference logs, detecting anomalies, and optimizing performance.

## 🎯 Purpose

Real-world AI deployments generate massive logs. This project provides:
- **Log Parsing**: Extract structured metrics from various inference frameworks
- **Statistical Analysis**: Latency distributions, percentiles, outliers
- **Anomaly Detection**: Identify performance regressions and failures
- **Visualization**: Interactive dashboards for exploration
- **Alerting**: Rule-based and ML-based alerting

## 📁 Project Structure

```
Inference-Log-Analytics/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── onnx_runtime.py
│   │   ├── tensorflow_serving.py
│   │   ├── triton.py
│   │   └── custom.py
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── latency.py
│   │   ├── throughput.py
│   │   ├── errors.py
│   │   └── anomaly.py
│   ├── visualizers/
│   │   ├── __init__.py
│   │   ├── dashboard.py
│   │   └── plots.py
│   └── alerts/
│       ├── __init__.py
│       └── rules.py
├── notebooks/
│   └── exploration.ipynb
├── configs/
│   └── default.yaml
└── tests/
    └── test_parsers.py
```

## 🚀 Quick Start

```python
from inference_analytics import LogAnalyzer

# Analyze ONNX Runtime logs
analyzer = LogAnalyzer.from_file("inference.log", format="onnx_runtime")
report = analyzer.analyze()

print(f"P50 Latency: {report.latency.p50:.2f}ms")
print(f"P99 Latency: {report.latency.p99:.2f}ms")
print(f"Error Rate: {report.errors.rate:.2%}")
```

## 📊 Key Metrics Extracted

| Metric | Description |
|--------|-------------|
| Latency (P50/P90/P99/P99.9) | End-to-end inference time |
| Queue Time | Time waiting before execution |
| Execution Time | Actual compute time |
| Throughput | Inferences per second |
| Error Rate | Failed inferences percentage |
| Memory Usage | Peak and average GPU memory |
| Batch Size Distribution | How requests are batched |

## 📈 Example Analysis

```
┌────────────────────────────────────────────────────┐
│           Latency Distribution                     │
│                                                    │
│  Count                                             │
│    │                    ████                       │
│    │                 ████████                      │
│    │              ███████████                      │
│    │           ██████████████                      │
│    │       ████████████████████                    │
│    │   ███████████████████████████  ▪▪            │
│    └────────────────────────────────────── ms      │
│        10   20   30   40   50   60   70+          │
│                                                    │
│  Stats: P50=25ms, P90=42ms, P99=58ms, P99.9=95ms │
└────────────────────────────────────────────────────┘
```

## 🔧 Supported Log Formats

- **ONNX Runtime**: Native profiling output
- **TensorFlow Serving**: Request logs
- **NVIDIA Triton**: Inference server logs
- **Custom**: Configurable JSON/CSV parsers

## License

MIT
