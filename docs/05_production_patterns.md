# 05. Production Patterns: Graphs & Batching

## Introduction
So far, we've composed services manually. For production, Ray offers the **Deployment Graph API** (better DAGs) and **Request Batching** (crucial for GPU efficiency).

## The Code: `production_graph.py`

```python
import ray
from ray import serve
from fastapi import FastAPI
import asyncio

app = FastAPI()

# COMPONENT 1: Batched Heavy Model
@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
class BatchModel:
    # THE BATCH DECORATOR
    # This automatically aggregates incoming individual requests into a list.
    # max_batch_size=4: Wait for up to 4 requests.
    # batch_wait_timeout_s=1.0: Or process whatever we have after 1s.
    @serve.batch(max_batch_size=4, batch_wait_timeout_s=1.0)
    async def __call__(self, requests: list):
        # This transparently aggregates concurrent requests into a list.
        
        # In a real ML model, you would stack these into a Tensor:
        # batch_tensor = torch.stack([r.tensor for r in requests])
        # result = model(batch_tensor)
        
        results = []
        for r in requests:
            results.append(f"Processed batch item: {r}")
            
        return results

# COMPONENT 2: Ingress
@serve.deployment
@serve.ingress(app)
class Ingress:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    @app.post("/")
    async def serve(self, data: str):
        # We send a single item. The BatchModel receives a list.
        return await self.model_handle.remote(data)

# BUILD THE GRAPH
model = BatchModel.bind()
ingress = Ingress.bind(model)
```

## Production Config: `config.yaml`

Instead of running python scripts in prod, we export a declarative config:

```bash
serve build production_graph:ingress -o config.yaml
```

This generates a YAML describing your entire application:
- Runtime Environments (pip packages)
- Import Paths
- Scaling Configs (replicas, CPUs)

You then deploy this "Spec" to the cluster:

```bash
serve run config.yaml
```

## Key Concepts

### Request Batching
GPUs are massive parallel processors. Sending 1 image at a time is wasteful. Sending 32 images at once takes roughly the same time as 1.
Ray Serve's `@serve.batch` transparently handles this coalescence.
*   **Technical Context**: This is essentially **Nagle's Algorithm** for application-layer requests.
    *   *Nagle's Algorithm (TCP)*: Improves efficiency by buffering small packets and sending them as a group, reducing protocol overhead.
    *   *Ray Batching*: Buffers individual inference requests (which are computationally expensive to process one-by-one due to GPU kernel launch overhead) and processes them as a single large tensor operation. It trades latency (waiting for the batch to fill) for massive throughput.

### Declarative Deployment
Moving from `python my_script.py` to `serve run config.yaml` is the shift from "Imperative" to "Declarative". This YAML is what lives in your Git repo and gets synced by ArgoCD.
