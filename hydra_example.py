import ray
from ray import serve
from fastapi import FastAPI
from omegaconf import OmegaConf, DictConfig
import hydra
import os

app = FastAPI()

# We need a way to find the config path relative to this file
# regardless of where we run 'serve run' from.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "conf")

@serve.deployment
@serve.ingress(app)
class HydraService:
    def __init__(self, user_config: dict):
        # 1. Load the Base Config using Hydra/OmegaConf
        # We manually compose the config because Serve controls the lifecycle,
        # not the @hydra.main decorator.
        with hydra.initialize_config_dir(config_dir=CONFIG_PATH, version_base=None):
            # Load 'config.yaml'
            self.base_cfg = hydra.compose(config_name="config")

        # 2. Merge with Ray Serve's user_config
        # This allows us to override values via Serve's config.yaml or dashboard
        if user_config:
            runtime_overrides = OmegaConf.create(user_config)
            self.cfg = OmegaConf.merge(self.base_cfg, runtime_overrides)
        else:
            self.cfg = self.base_cfg

        # 3. Initialize Component
        self.message = self.cfg.app.message
        self.repeat = self.cfg.app.repeat

    def reconfigure(self, user_config: dict):
        # Ray Serve calls this method when 'user_config' updates dynamically.
        if user_config:
            runtime_overrides = OmegaConf.create(user_config)
            self.cfg = OmegaConf.merge(self.base_cfg, runtime_overrides)
            self.message = self.cfg.app.message
            self.repeat = self.cfg.app.repeat
            print(f"Reconfigured! Message: {self.message}")

    @app.get("/")
    def get_message(self):
        return {"result": self.message * self.repeat}

# For local testing with `python hydra_example.py`
if __name__ == "__main__":
    # Mock a serve deployment
    deployment = HydraService.bind({})
    serve.run(deployment)
    import requests
    print(requests.get("http://localhost:8000/").json())
