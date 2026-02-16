# 04. Scaling & FastAPI: Advanced Inference

## Introduction
A single Python process is limited by the Global Interpreter Lock (GIL). To handle high traffic, you need to scale horizontally. 
Ray Serve makes this trivial compared to Kubernetes, where you'd need to configure ReplicaSets, Services, and Ingresses.

## The Code: `advanced_pipeline.py`

```python
import ray
from ray import serve
from fastapi import FastAPI
import time

# 1. Integrate FastAPI
# Instead of a raw class, we use FastAPI for better routing, validation, and docs.
app = FastAPI()

@serve.deployment(num_replicas=2)  # <--- SCALING MAGIC
@serve.ingress(app)
class HeavyModel:
    @app.get("/predict")
    def predict(self):
        # Simulate CPU work
        time.sleep(0.1)
        # pid identifying which replica handled the request
        import os
        return {"result": "processed", "pid": os.getpid()}

    @app.get("/health")
    def health(self):
        return {"status": "ok"}

# Bind the deployment
deployment = HeavyModel.bind()

if __name__ == "__main__":
    serve.run(deployment)
    # The endpoints are now available at http://localhost:8000/predict
```

## Key Concepts

### `num_replicas=2`
This single line tells Ray to:
1.  Start **two independent processes** (Actors) of the `HeavyModel` class.
2.  Load balance traffic between them.
*   **Technical Context**: Ray schedules two separate actor processes. The incoming traffic is load-balanced across these processes by the Serve Proxy.

### FastAPI Integration
Ray Serve is ASGI compatible. You can take *any* ASGI app (FastAPI, Django, etc.) and wrap it in `@serve.ingress`.
*   **Routing**: Ray strips the deployment prefix and forwards the rest of the path to FastAPI.

### Routing & Load Balancing
When a request hits the Proxy (port 8000), Ray checks the routing table.
*   If multiple replicas exist, it uses a **random** (default) or **power-of-two-choices** strategy to pick a replica.
*   This happens entirely within the Ray cluster, with no external Load Balancer needed.

## Why this is powerful
In K8s, scaling often means spinning up new Pods, which is slow (pull image, start container).
In Ray, scaling is spinning up a new Process on an existing node, which is milliseconds-fast (if resources are available).
