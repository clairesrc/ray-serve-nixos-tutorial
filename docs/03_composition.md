# 03. Composition: Building Pipelines

## Introduction
Real ML applications are rarely a single function. They are pipelines:
`Preprocess` -> `Model Inference` -> `Postprocess`.

In Kubernetes, you might stick these in separate containers and talk via HTTP (Sidecars). Ray Serve allows you to compose them programmatically using **Ray Object Refs** (Futures) and **Handles**.

## The Code: `pipeline.py`

```python
import ray
from ray import serve
import requests

# COMPONENT 1: The Model
# This represents a heavy computation service.
@serve.deployment
class Model:
    def __call__(self, data: str):
        return f"Model processed: {data}"

# COMPONENT 2: The Ingress (Orchestrator)
# This handles the HTTP request and delegates work to the Model.
@serve.deployment
@serve.ingress(app) # Optional if we use FastAPI, but good for raw classes
class Ingress:
    # Dependency Injection!
    # The Ingress takes a 'model_handle' as an argument.
    def __init__(self, model_handle):
        self.model_handle = model_handle

    async def __call__(self, request):
        data = await request.body()
        
        # RPC Call (Remote Procedure Call)
        # self.model_handle.remote(...) sends a request to the Model deployment.
        # It returns a RayObjectRef (Future).
        ref = await self.model_handle.remote(data.decode("utf-8"))
        
        # We await the result. Ray handles the networking / serialization.
        return await ref

# WIRING IT TOGETHER
# 1. Define the Model config (lazy).
model = Model.bind()

# 2. Inject Model into Ingress.
app = Ingress.bind(model)

if __name__ == "__main__":
    serve.run(app)
    print(requests.post("http://localhost:8000/", data="Hello Pipeline").text)
```

## Key Concepts

### `bind()` as Dependency Injection
Notice `Ingress.bind(model)`. This is declarative. We are building a **Deployment Graph**.

*   **Technical Context**: `bind()` lazily builds a DAG (Directed Acyclic Graph) of deployments. Ray Serve resolves this graph at runtime to instantiate the necessary actors and establish routing.

### The `ServeHandle`
When `Ingress` receives `model_handle`, it gets a specialized RPC client.
*   `model_handle.remote(data)`: Asynchronous call. Non-blocking.
*   **Load Balancing**: If `Model` had 10 replicas, Ray would verify round-robin or power-of-two-choices load balancing automatically for this call.
*   **Locality**: If a `Model` replica is on the same node, Ray uses shared memory (Zero-Copy) to pass data. This is much faster than HTTP over localhost.

## Why this matters for ML
ML artifacts (images, tensors) are large. Passing them via HTTP JSON is slow. Ray uses a specialized object store called **Plasma**.
*   **Plasma (Shared Memory)**: A high-performance shared-memory object store.
*   **Zero-Copy**: When two actors on the same node need to share a large tensor, Ray maps the memory region into both processes. Data is not copied. This achieves near-infinite bandwidth for local IPC.
