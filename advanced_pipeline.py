import ray
from ray import serve
from fastapi import FastAPI
import time
import requests

ray.init()

app = FastAPI()

@serve.deployment(num_replicas=2)
class HeavyModel:
    def __call__(self, data: str) -> str:
        # Simulate heavy processing
        time.sleep(0.5)
        return f"HeavyModel processed: {data}"

@serve.deployment
@serve.ingress(app)
class FastAPIIngress:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    @app.post("/predict")
    async def predict(self, data: str):
        result = await self.model_handle.remote(data)
        return {"result": result}

    @app.get("/health")
    def health(self):
        return {"status": "ok"}

# Bind the deployments
model = HeavyModel.bind()
ingress = FastAPIIngress.bind(model)

if __name__ == "__main__":
    serve.run(ingress)
    
    print("\n--- Testing Health Endpoint ---")
    print(requests.get("http://localhost:8000/health").json())

    print("\n--- Testing Predict Endpoint ---")
    start = time.time()
    # Send a few requests to see them being handled
    for i in range(3):
        resp = requests.post("http://localhost:8000/predict", params={"data": f"Request {i}"})
        print(f"Response {i}: {resp.json()}")
    
    print(f"\nTotal time: {time.time() - start:.2f}s")
    print("\nYou can also view the auto-generated docs at http://localhost:8000/docs")
