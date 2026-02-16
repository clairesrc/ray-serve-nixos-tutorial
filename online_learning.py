import ray
from ray import serve
from fastapi import FastAPI
import torch
import torch.nn as nn
import torch.optim as optim
import requests

ray.init()

app = FastAPI()

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple linear regression: y = wx + b
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

@serve.deployment
@serve.ingress(app)
class OnlineLearner:
    def __init__(self):
        self.model = SimpleModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion = nn.MSELoss()
        
        # Initialize weights to known values for demonstration
        with torch.no_grad():
            self.model.linear.weight.fill_(1.0)
            self.model.linear.bias.fill_(0.0)

    @app.post("/predict")
    def predict(self, x: float):
        # Forward pass (thread-safe in terms of model state read if no training happens concurrently)
        # In Ray Serve, a single replica processes requests. 
        # By default, concurrency is 1 if methods are not async.
        # If async, we might need Locking. Here we use sync methods for simplicity/safety.
        input_tensor = torch.tensor([[x]])
        with torch.no_grad():
            output = self.model(input_tensor)
        return {"prediction": output.item()}

    @app.post("/train")
    def train(self, x: float, y: float):
        # Perform one step of backprop
        self.optimizer.zero_grad()
        input_tensor = torch.tensor([[x]])
        target_tensor = torch.tensor([[y]])
        
        output = self.model(input_tensor)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "new_weight": self.model.linear.weight.item(),
            "new_bias": self.model.linear.bias.item()
        }

# Bind the deployment
learner = OnlineLearner.bind()

if __name__ == "__main__":
    serve.run(learner)
    
    print("\n--- Initial State ---")
    # Target function: y = 2x + 1
    # Initial model:   y = 1x + 0
    
    # Predict at x=1. Expect 1.0 (1*1 + 0)
    resp = requests.post("http://localhost:8000/predict", params={"x": 1.0})
    print(f"Prediction for x=1: {resp.json()}")

    print("\n--- Training Step ---")
    # Train heavily on x=1, y=3 (pushing towards y=2x+1)
    # Perform a few updates
    for i in range(5):
        resp = requests.post("http://localhost:8000/train", params={"x": 1.0, "y": 3.0})
        print(f"Train Step {i+1}: {resp.json()}")

    print("\n--- After Training ---")
    # Predict at x=1. Expect closer to 3.0
    resp = requests.post("http://localhost:8000/predict", params={"x": 1.0})
    print(f"Prediction for x=1: {resp.json()}")
