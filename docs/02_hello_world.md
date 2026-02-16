# 02. Hello World: The Basic Deployment

## Introduction
In this section, we deploy our first "Microservice" using Ray Serve. In Ray terminology, a service is called a **Deployment**.

## The Code: `hello.py`

```python
import ray
from ray import serve
import requests

# 1. Initialize Ray.
# On a laptop, this starts a mini-cluster locally (Head + Worker in one).
# In prod, this connects to an existing K8s cluster.
ray.init()

# 2. Define a Deployment.
# The @serve.deployment decorator turns a standard Python class into a managed service.
# This registers the class with the Ray Serve Controller, which manages its lifecycle and scheduling.
@serve.deployment
class HelloWorld:
    # 3. Request Handling.
    # The __call__ method is the entrypoint for HTTP requests.
    # 'request' is a Starlette Request object (standard async Python web request).
    def __call__(self, request):
        return "Hello world!"

# 4. Bind the deployment.
# This creates the "Application" object. It doesn't run it yet.
# It effectively defines the dependency graph (DAG) of your services.
app = HelloWorld.bind()

if __name__ == "__main__":
    # 5. Run the application.
    # This deploys the DAG to the Ray cluster.
    # It allocates resources (CPUs/RAM), starts the HTTP Proxy, and registers routes.
    serve.run(app)
    
    # 6. Verify.
    # By default, Ray Serve exposes HTTP on localhost:8000.
    print(requests.get("http://localhost:8000/").text)
```

## Key Concepts

### `@serve.deployment`
This is your **Infrastructure-as-Code**. It tells Ray:
"I want to run this class as a long-running service."
You can pass arguments here like `num_replicas=3`, `ray_actor_options={"num_gpus": 1}`.

### The Actor Model
Unlike a stateless function (e.g., AWS Lambda), this class is instantiated once and persists in memory.
*   **Technical Context**: This is the **Actor Model**. The process remains alive, maintaining its state (heap memory) across multiple incoming requests. This is crucial for ML because loading models into GPU memory is expensive and slow.

### `serve.run()`
This performs the deployment.
1.  Starts a **Proxy** (like Nginx/Envoy) on port 8000.
2.  Starts the **Controller** (deployment manager).
3.  Starts **Replicas** of your `HelloWorld` class.
4.  Updates the routing table so `GET /` goes to `HelloWorld`.

## Execution

Run it inside your Nix shell:

```bash
python hello.py
```
