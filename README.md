# Ray Serve for Software Engineers: A Tutorial Series

Welcome. This repository contains a comprehensive, line-by-line tutorial series designed for **Software Engineers** (running NixOS) who are interested in **Machine Learning**.

We provide a technical mapping of Distributed ML concepts (Actors, Tensors, Training) to their systems engineering counterparts (Processes, Shared Memory, State Mutation).

## The Series

1.  [**Environment Setup**](docs/01_environment_setup.md): Nix Flakes & Python Dependencies.
2.  [**Hello World**](docs/02_hello_world.md): The Actor Model & Basic Deployments.
3.  [**Composition**](docs/03_composition.md): Service-to-Service RPC & Pipelines.
4.  [**Scaling & FastAPI**](docs/04_scaling_fastapi.md): Replicas, Load Balancing, & Gateways.
5.  [**Production Patterns**](docs/05_production_patterns.md): Batching, DAGs, & Config Management.
6.  [**Stateful ML**](docs/06_stateful_ml.md): Online Learning, Tensors, & Backpropagation.
7.  [**Concurrency**](docs/07_concurrency.md): AsyncIO, Locking, & Throughput.
8.  [**Clustering**](docs/08_clustering.md): Head Nodes, Worker Nodes, & Resources.
9.  [**Kubernetes Deployment**](docs/09_k8s_deployment.md): KubeRay Operator & CRDs.
10. [**Observability**](docs/10_observability.md): Metrics, Prometheus, & Logging.
11. [**Config Management**](docs/11_config_management.md): Hydra, OmegaConf & Structured Configs.
12. [**Conclusion**](docs/12_conclusion.md): Recap & Advanced Topics (LLMs, GenAI).

## Quick Start

Enter the environment:
```bash
nix develop
```

Run the "Production Graph" example:
```bash
serve run config.yaml
```

Run the "Online Learning" example:
```bash
python online_learning_locking.py
```
