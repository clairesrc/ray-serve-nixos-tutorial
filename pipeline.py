import ray
from ray import serve
import requests
from starlette.requests import Request

ray.init()

@serve.deployment
class Model:
    def __call__(self, data: str) -> str:
        return f"Model received: {data}"

@serve.deployment
class Ingress:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    async def __call__(self, request: Request) -> str:
        data = await request.body()
        data_str = data.decode("utf-8")
        # Call the model deployment
        result = await self.model_handle.remote(data_str)
        return f"Ingress processed result: {result}"

# Bind the deployments
model = Model.bind()
ingress = Ingress.bind(model)

if __name__ == "__main__":
    serve.run(ingress)
    resp = requests.post("http://localhost:8000/", data="Hello Pipeline!")
    print(resp.text)
