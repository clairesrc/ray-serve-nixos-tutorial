# 06. Stateful ML: Online Learning

## Introduction
Use cases like "RecSys" (Recommendation Systems) or "Fraud Detection" often require **Online Learning**: the model learns from new data in real-time.

This requires **State**. In a typical K8s microservice, pods are stateless. State lives in Redis/Postgres.
In Ray, **Actors are Stateful**. They hold the model weights in RAM and update them given new data.

## The Code: `online_learning.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from ray import serve
from fastapi import FastAPI

app = FastAPI()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple linear equation: y = wx + b
        # "Weights" are just numbers we are trying to optimize.
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

@serve.deployment
@serve.ingress(app)
class OnlineLearner:
    def __init__(self):
        self.model = SimpleModel()
        # The Optimizer decides *how* to change weights to reduce error.
        # SGD = Stochastic Gradient Descent.
        # lr=0.1 = Learning Rate. Think of this as the "Step Size".
        #   - Too small: Learning takes forever.
        #   - Too big: We overshoot the target and become unstable.
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        
        # The Loss Function measures *how wrong* a prediction was.
        self.criterion = nn.MSELoss()

    @app.post("/predict")
    def predict(self, x: float):
        # INFERENCE (Read-Only)
        # We don't track gradients here (no_grad) to save memory.
        with torch.no_grad():
            output = self.model(torch.tensor([[x]]))
        return {"prediction": output.item()}

    @app.post("/train")
    def train(self, x: float, y: float):
        # TRAINING (Write)
        
        # 1. Reset gradients (buffer)
        self.optimizer.zero_grad()
        
        # 2. Forward pass (Make a guess)
        output = self.model(torch.tensor([[x]]))
        
        # 3. Calculate Loss (How bad was the guess?)
        loss = self.criterion(output, torch.tensor([[y]]))
        
        # 4. Backward Pass (Calculus magic)
        # Compute the "Gradient": The direction of steepest ascent of the error.
        # Ideally, we want to go the opposite way (Descent) to minimize error.
        # This stores the required change in .grad attribute of every weight.
        loss.backward()
        
        # 5. Step (Update)
        # Apply the update: weight = weight - (learning_rate * gradient)
        # THIS MUTATES THE STATE OF THE CLASS
        self.optimizer.step()
        
        return {"loss": loss.item()}
```

## Key Concepts

### Model Weights as "State"
Think of `self.model` as an in-memory database.
-   `/predict` is a **Read Query**.
-   `/train` is a **Write Transaction**.

### The Training Loop (Simplified)

To explain this to a non-ML engineer: Imagine a cannon trying to hit a target.
1.  **Forward (The Shot)**: We fire the cannon (Predict) and see where it lands.
2.  **Loss (The Measurement)**: We measure the distance between the landing spot and the target.
3.  **Backward (The Calculation)**: We calculate how much to adjust the angle and powder charge to hit closer next time.
4.  **Optimizer (The Adjustment)**: We physically turn the knobs to update the cannon's settings (Weights).

### Why Ray?
Doing this in a stateless K8s pod is hard. You'd need to fetch weights from a Parameter Server (Redis), update them, and push them back for *every request*.
Ray keeps the weights locally in the Actor's heap memory. Updates are just in-memory variable assignments (`self.weight += delta`).
*   **Contrast with Stateless**: If this were a stateless Lambda/Pod, you would have to fetch the weights from Redis (Network I/O), update them, and save them back to Redis (Network I/O) for *every single training step*.
*   **Performance**: Local memory updates are nanoseconds. Network round-trips are milliseconds. For training loops running thousands of times per second, this statefulness is mandatory.
