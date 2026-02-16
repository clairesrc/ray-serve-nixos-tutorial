import ray
from ray import serve
from starlette.requests import Request
from typing import List
import time
import asyncio

@serve.deployment
class BatchModel:
    def __init__(self):
        self.model_loaded = True

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=1.0)
    async def handle_batch(self, inputs: List[str]) -> List[str]:
        # Simulate batched inference (e.g., passing a batch to a GPU model)
        print(f"Processing batch of size {len(inputs)}: {inputs}")
        time.sleep(1.0) # Simulate heavy work
        return [f"Processed({i})" for i in inputs]

    async def __call__(self, input_item: str) -> str:
        # The __call__ method now delegates to the batched handler
        # serve.batch automatically aggregates calls to this method into handle_batch
        return await self.handle_batch(input_item)

@serve.deployment
class Ingress:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    async def __call__(self, request: Request) -> str:
        data = await request.body()
        data_str = data.decode("utf-8")
        
        # Determine if we should send a burst of requests to test batching
        if data_str == "BATCH_TEST":
            # Simulate concurrent requests coming in
            tasks = [self.model_handle.remote(f"Item {i}") for i in range(4)]
            results = await asyncio.gather(*tasks)
            return f"Batch result: {results}"
        
        # Normal single request
        result = await self.model_handle.remote(data_str)
        return f"Result: {result}"

# Deployment Graph API
# We allow the user to build this graph and export it to YAML
model = BatchModel.bind()
app = Ingress.bind(model)
