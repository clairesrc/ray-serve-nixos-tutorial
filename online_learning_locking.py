import ray
from ray import serve
from fastapi import FastAPI
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
import requests

# ray.init(): Connects to an existing Ray cluster or starts a new local one.
ray.init()

# FastAPI: A modern web framework for building APIs with Python.
# This app will define the HTTP routes for our deployment.
app = FastAPI()

# define a simple PyTorch model class inheriting from nn.Module.
# nn.Module is the base class for all neural network modules in PyTorch.
# It handles tracking of parameters (weights) and sub-modules.
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(in_features=1, out_features=1, bias=True)
        # Applies a linear transformation to the incoming data: y = xA^T + b.
        # Here, it learns a simple 1D linear relationship: y = weight * x + bias.
        self.linear = nn.Linear(1, 1)

    # The forward pass defines how the input data flows through the network layers.
    def forward(self, x):
        # Pass input x through the linear layer.
        return self.linear(x)

# @serve.deployment: Decorator that turns a Python class into a Ray Serve Deployment.
# This manages the lifecycle, scaling, and request handling for the class.
@serve.deployment
# @serve.ingress(app): Wraps the FastAPI app 'app' inside the deployment.
# This allows us to use FastAPI decorators (@app.post) to define routes.
@serve.ingress(app)
class AsyncOnlineLearner:
    def __init__(self):
        # Instantiate our model.
        self.model = SimpleModel()
        
        # optim.SGD(params, lr): Stochastic Gradient Descent optimizer.
        # It takes the model's parameters (weights/biases) and a learning rate (lr).
        # It will update the parameters based on the gradients computed during backprop.
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        
        # nn.MSELoss(): Mean Squared Error Loss function.
        # Computes the average squared difference between the predicted output and the target value.
        # Commonly used for regression tasks.
        self.criterion = nn.MSELoss()
        
        # Initialize weights to known values for demonstration clarity.
        # torch.no_grad(): Context manager that disables gradient calculation.
        # We don't want these manual initialization operations to be tracked in the computational graph.
        with torch.no_grad():
            self.model.linear.weight.fill_(1.0) # Set weight 'w' to 1.0
            self.model.linear.bias.fill_(0.0)   # Set bias 'b' to 0.0
            
        # asyncio.Lock(): A synchronization primitive for asyncio tasks.
        # Ensures that only one coroutine can execute a critical section at a time.
        # We need this to prevent concurrent 'train' calls from updating the model weights simultaneously,
        # or a 'predict' call reading weights while they are being updated.
        self.lock = asyncio.Lock()

    @app.post("/predict")
    async def predict(self, x: float):
        # async with self.lock: Acquire the lock before entering this block.
        # This ensures thread-safety (coroutine-safety) for accessing the model state.
        async with self.lock:
            # torch.tensor([[x]]): Create a 2D tensor from the input float x.
            # Shape (1, 1) corresponds to (batch_size=1, features=1).
            input_tensor = torch.tensor([[x]])
            
            # with torch.no_grad(): Disable gradient tracking during inference.
            # This reduces memory usage and speeds up computation since we don't need backprop here.
            with torch.no_grad():
                # self.model(input_tensor): Calls the model's forward() method.
                output = self.model(input_tensor)
            
            # output.item(): Convert the single-value tensor result back to a standard Python float.
            return {"prediction": output.item()}

    @app.post("/train")
    async def train(self, x: float, y: float):
        # Acquire lock to ensure exclusive access during the training update.
        async with self.lock:
            # 1. Zero the parameter gradients.
            # PyTorch accumulates gradients on subsequent backward passes by default.
            # We must clear them before starting a new optimization step.
            self.optimizer.zero_grad()
            
            input_tensor = torch.tensor([[x]])
            target_tensor = torch.tensor([[y]])
            
            # 2. Forward pass: Compute predicted output by passing input to the model.
            output = self.model(input_tensor)
            
            # 3. Compute loss: Calculate the error between prediction and target using MSELoss.
            loss = self.criterion(output, target_tensor)
            
            # 4. Backward pass: specific to Neural Networks.
            # loss.backward() computes the gradient of the loss with respect to all model parameters (weights/biases).
            # It traverses the graph backwards from the loss function.
            loss.backward()
            
            # 5. Optimization step: update weights.
            # self.optimizer.step() updates the model parameters based on the computed gradients and the learning rate rule.
            self.optimizer.step()
            
            return {
                "loss": loss.item(),
                "new_weight": self.model.linear.weight.item(),
                "new_bias": self.model.linear.bias.item()
            }

# Bind the deployment
learner = AsyncOnlineLearner.bind()

if __name__ == "__main__":
    serve.run(learner)
    
    print("\n--- Initial State ---")
    resp = requests.post("http://localhost:8000/predict", params={"x": 1.0})
    print(f"Prediction for x=1: {resp.json()}")

    print("\n--- Training Step ---")
    # Demonstrate functionality is preserved
    for i in range(5):
        resp = requests.post("http://localhost:8000/train", params={"x": 1.0, "y": 3.0})
        print(f"Train Step {i+1}: {resp.json()}")

    print("\n--- After Training ---")
    resp = requests.post("http://localhost:8000/predict", params={"x": 1.0})
    print(f"Prediction for x=1: {resp.json()}")
