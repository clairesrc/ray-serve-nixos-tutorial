# 10. Observability: Metrics & Monitoring

## Introduction
You can't manage what you can't measure. Ray Serve comes with built-in **Prometheus** metrics and structured **Logging**.

## Metrics

Ray exports metrics at `http://localhost:8080/metrics` (or `http://<HEAD_IP>:8080/metrics`).
These are standard Prometheus format text lines.

## Ray Dashboard
The **Ray Dashboard** (default: `http://localhost:8265`) is the primary Web UI for observability. It provides:
*   **Cluster Tab**: View node status, CPU/GPU usage, and object store memory.
*   **Serve Tab**: View all running applications, their routes, and per-deployment metrics.
*   **Logs**: Stream logs from every worker across the cluster into a single consolidated view.

> [!TIP]
> Use the Dashboard during development to debug why a deployment might be stuck or why a specific route is failing.

### Key Metrics to Watch

1.  **Traffic**:
    *   `serve_num_router_requests`: Total requests hitting the proxy.
    *   `serve_num_outstanding_requests`: Requests currently being processed.

2.  **Latency** (The Golden Signal):
    *   `serve_deployment_request_latency_ms`: End-to-end latency.
    *   `serve_deployment_queuing_latency_ms`: Time spent waiting for a replica. (High queuing = Scale up!).

3.  **Saturation**:
    *   `ray_node_cpu_utilization`: Physical resource usage.

## The Code: `metrics_client.py`

This script demonstrates how to scrape specific metrics programmatically:

```python
import requests

def print_metrics():
    resp = requests.get("http://localhost:8080/metrics")
    for line in resp.text.splitlines():
        if "serve_deployment_request_latency_ms" in line:
            print(line)

# This is useful for custom metric export or health checks.
```

## Logging

Print statements (`print("hello")`) in Ray Actors are captured and streamed to the Driver (your terminal) and to log files in `/tmp/ray/session_latest/logs/`.

For production, use the standard python logger:

```python
import logging
logger = logging.getLogger("ray.serve")

@serve.deployment
class MyService:
    def __call__(self):
        logger.info("Processing request", extra={"req_id": ...})
```


