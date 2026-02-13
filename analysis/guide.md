# Inference Log Analytics - Analysis Guides

## Step-by-Step Analysis Guide

### 1. Data Collection

First, ensure you're collecting the right metrics:

```python
# Recommended log format
import json
from datetime import datetime

def log_inference(request_id, model_name, latency_ms, batch_size, status):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "model": model_name,
        "latency_ms": latency_ms,
        "batch_size": batch_size,
        "status": status,
        "gpu_util": get_gpu_utilization(),
        "memory_used_gb": get_memory_usage()
    }
    print(json.dumps(log_entry))
```

### 2. Loading Logs

```python
from src.analyzer import InferenceLogAnalyzer

# Initialize analyzer
analyzer = InferenceLogAnalyzer()

# Load logs from file
analyzer.load_logs("data/production_logs.txt")

# Or load from directory
analyzer.load_logs_from_dir("data/logs/")
```

### 3. Basic Analysis

```python
# Generate summary statistics
summary = analyzer.summary()
print(f"Total requests: {summary['total_requests']}")
print(f"Average latency: {summary['avg_latency_ms']:.2f} ms")
print(f"P99 latency: {summary['p99_latency_ms']:.2f} ms")
print(f"Error rate: {summary['error_rate']:.2%}")
```

### 4. Anomaly Detection

```python
# Find outliers
outliers = analyzer.detect_anomalies(
    metric="latency_ms",
    method="zscore",
    threshold=3.0
)

print(f"Found {len(outliers)} anomalous requests")
for outlier in outliers[:5]:
    print(f"  Request {outlier['request_id']}: {outlier['latency_ms']}ms")
```

### 5. Trend Analysis

```python
# Analyze trends over time
trends = analyzer.analyze_trends(
    metric="latency_ms",
    window="1h",
    aggregation="mean"
)

# Plot the trend
import matplotlib.pyplot as plt
plt.plot(trends['timestamp'], trends['value'])
plt.xlabel('Time')
plt.ylabel('Average Latency (ms)')
plt.title('Latency Trend')
plt.show()
```

### 6. Correlation Analysis

```python
# Find correlations between metrics
correlations = analyzer.correlate_metrics(
    metrics=["latency_ms", "batch_size", "gpu_util", "memory_used_gb"]
)

print("Correlation matrix:")
print(correlations)
```

### 7. Export Reports

```python
# Generate HTML report
analyzer.generate_report(
    output="reports/daily_report.html",
    format="html"
)

# Export to CSV for further analysis
analyzer.export_csv("reports/data_export.csv")
```

## Common Analysis Patterns

### Latency Spike Investigation

```python
def investigate_spike(analyzer, start_time, end_time):
    # Filter to time window
    spike_data = analyzer.filter_time_range(start_time, end_time)
    
    # Check concurrent requests
    print(f"Requests during spike: {len(spike_data)}")
    print(f"Average batch size: {spike_data['batch_size'].mean():.1f}")
    print(f"Max concurrent: {spike_data.groupby('timestamp').size().max()}")
    
    # Check resource utilization
    print(f"GPU utilization: {spike_data['gpu_util'].mean():.1f}%")
    print(f"Memory usage: {spike_data['memory_used_gb'].mean():.2f} GB")
```

### Model Comparison

```python
def compare_models(analyzer, model_names):
    results = {}
    for model in model_names:
        model_data = analyzer.filter_model(model)
        results[model] = {
            "avg_latency": model_data['latency_ms'].mean(),
            "p99_latency": model_data['latency_ms'].quantile(0.99),
            "throughput": len(model_data) / model_data['duration'].sum(),
        }
    return pd.DataFrame(results).T
```
