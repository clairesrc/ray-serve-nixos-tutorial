# 07. Concurrency: High Throughput & Locking

## Introduction
In the previous chapter, we mutated state (`self.model`). But what happens when 100 requests arrive at once?
Ray Actors are single-threaded by default. But if you use `async def`, they become **Concurrent**. This is great for throughput (handling I/O) but dangerous for state.

## The Code: `online_learning_locking.py`

```python
import asyncio
from ray import serve
from fastapi import FastAPI
# ... imports ...

@serve.deployment
@serve.ingress(app)
class AsyncOnlineLearner:
    def __init__(self):
        self.model = SimpleModel()
        # ... init optimizer ...
        
        # THE LOCK
        # Required because 'async def' runs on an event loop.
        # While one coroutine awaits I/O, others can run. We need to prevent
        # race conditions on the shared model state.
        self.lock = asyncio.Lock()

    @app.post("/predict")
    async def predict(self, x: float):
        # We lock here to ensure we don't read the weights *while* they are being updated.
        async with self.lock:
            # ... inference code ...
            pass

    @app.post("/train")
    async def train(self, x: float, y: float):
        async with self.lock:
            # CRITICAL SECTION
            # We must finish the entire Forward -> Backward -> Step sequence
            # without interruption to keep the math correct.
            self.optimizer.zero_grad()
            # ...
            loss.backward()
            self.optimizer.step()
```

## Key Concepts

### Sync vs Async Actors
*   **Sync (`def`)**: Ray executes 1 request at a time per replica. Safe but slow if you do I/O.
*   **Async (`async def`)**: Ray runs an Event Loop. Multiple requests can be "in flight".
    *   **Throughput**: Massive increase for I/O bound workloads.
    *   **Risk**: You must protect shared state (`self.model`).

### Why Locking?
In a specialized ML system, you often want **Reader/Writer locks**:
*   Thousands of `/predict` calls (Readers) can run concurrently.
*   One `/train` call (Writer) needs exclusive access.
*   (Our example uses a simple Mutex for simplicity).

### Technical Context
This follows standard concurrency patterns. The Actor is a single process with an event loop. Adding `async` enables cooperative multitasking. Synchronization primitives (Locks, Semaphores) are required to protect critical sections that mutate shared state.
