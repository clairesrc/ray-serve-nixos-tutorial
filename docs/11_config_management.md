# 11. Configuration Management: Hydra & OmegaConf

## Introduction
Production ML systems have two types of configuration:
1.  **Infrastructure Config**: Replicas, CPUs, GPUs, Autoscaling rules. (Managed by **Ray Serve**).
2.  **Application Config**: Hyperparameters, Feature flags, S3 paths, Thresholds. (Managed by **Hydra**).

Mixing these is messy. This chapter shows how to keep them separate but integrated.

## The Pattern: Composition
We want **Ray Serve** to handle the deployment lifecycle and **Hydra** to handle the application logic.

### 1. Define Application Config
Use Hydra's structured config or YAML files.

`conf/config.yaml`:
```yaml
app:
  message: "Hello form Hydra!"
  repeat: 1

ml:
  model_name: "resnet50"
  threshold: 0.85
```

### 2. The Deployment Wrapper
In your deployment class, load the Hydra config during `__init__`.

```python
import hydra
from omegaconf import OmegaConf

@serve.deployment
class HydraService:
    def __init__(self, user_config: dict):
        # 1. Load Base Config (Hydra)
        with hydra.initialize(config_path="conf"):
            self.base_cfg = hydra.compose(config_name="config")
        
        # 2. Merge with Runtime Overrides (Ray Serve)
        # Ray passes the 'user_config' dict from serve.run() or the dashboard.
        if user_config:
            runtime_overrides = OmegaConf.create(user_config)
            self.cfg = OmegaConf.merge(self.base_cfg, runtime_overrides)
        else:
            self.cfg = self.base_cfg
            
    def reconfigure(self, user_config: dict):
        # Ray calls this when you update the config without restarting the actor!
        if user_config:
            self.cfg = OmegaConf.merge(self.base_cfg, OmegaConf.create(user_config))
```

## Running the Example

We have provided `hydra_example.py` in the repo.

1.  **Run it**:
    ```bash
    python hydra_example.py
    ```

2.  **Verify**:
    The service initializes with the defaults from `conf/config.yaml`.

## Dynamic Updates
Because we implemented `reconfigure()`, you can update the application config *at runtime* via the Ray Dashboard or the Serve REST API without restarting the heavy model actors.

This effectively gives you a **Dynamic Feature Flag** system for free.
