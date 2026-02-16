# 01. Environment Setup

## Introduction
As a Software Engineer, you know that reproducible environments are the bedrock of stable systems. In the Python/ML ecosystem, this is often a pain point due to conflicting pip dependencies and system-level library requirements (like CUDA).

Since you are running **NixOS**, we will leverage **Nix Flakes** to create a hermetic, reproducible development environment. This replaces the traditional `venv` + `requirements.txt` or `Dockerfile` workflow for local development.

## The Configuration: `flake.nix`

We use a `flake.nix` to define our dev shell. Here is the breakdown of the one we used:

```nix
{
  description = "Ray Serve Tutorial Environment";

  inputs = {
    # We pin nixpkgs to a specific commit to ensure reproducibility.
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # We target x86_64-linux.
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # Python 3.11+ is recommended for modern ML libraries.
          (python3.withPackages (ps: with ps; [
            # CORE DEPENDENCY
            ray

            # SERVE DEPENDENCIES
            # (In pip, these are installed via 'ray[serve]'. In Nix, we list them explicitly.)
            starlette
            uvicorn
            fastapi
            requests

            # ML DEPENDENCIES
            # Torch is our numerical engine. In platform terms, think of it
            # as a library that does matrix math very fast, sometimes on GPUs.
            torch
            
            # UTILS
            # 'watchfiles' allows Ray to hot-reload code when you change files.
            watchfiles 
            # DASHBOARD DEPENDENCIES
            # Ray Dashboard requires a significant stack for its UI and metrics.
            aiohttp
            aiohttp-cors
            opencensus
            prometheus-client
            opentelemetry-api
            psutil
            msgpack
          ]))
        ];

        # Environment variables for the shell
        shellHook = ''
          echo "Welcome to the Ray Serve Tutorial Environment!"
          echo "Python version: $(python --version)"
          echo "Ray version: $(python -c 'import ray; print(ray.__version__)')"
        '';
      };
    };
}
```

## Key Dependencies Explained

### `ray`
The distributed computing primitive. It provides the "Actor" model (stateful workers) and "Task" model (stateless functions).
*   **Technical Context**: Think of Ray as a distributed scheduler that manages Python processes across a cluster, handling serialization and communication.
    *   **Serialization (CloudPickle)**: Standard Python `pickle` fails on lambdas and nested functions. Ray uses `cloudpickle` to serialize code (closures, classes) so you can ship functions to remote workers.
    *   **Communication**: Uses gRPC for control and Shared Memory for data.

### `torch` (PyTorch)
The de facto standard library for deep learning.
*   **Technical Context**: A high-performance numerical computation library. It provides:
    1.  **Tensors**: N-dimensional arrays (similar to NumPy ndarrays) but with **Hardware Acceleration**.
        *   *Context*: Deep Learning relies heavily on Matrix Multiplication. GPUs (and TPUs) are specialized circuits that execute these operations in massive parallel streams, offering 10-100x speedups over CPUs.
    2.  **Autograd**: Automatic differentiation engine for **Gradient-Based Optimization**.
        *   *Context*: Training a neural network is essentially finding the minimum value of a function (the error). "Gradients" tell us the direction of steepest ascent. We move in the opposite direction (descent) to minimize error. PyTorch calculates these derivatives automatically.
    3.  **nn.Module**: Base class for all neural network modules.

### `fastapi` / `uvicorn`
Ray Serve wraps these industry-standard web tools.
*   **Technical Context**: Ray Serve mounts FastAPI apps as ingress points. It handles the HTTP termination and proxying, while FastAPI handles the routing and validation within the Python worker process.

## Usage

To enter this environment, simply run:

```bash
nix develop
```

This drops you into a bash shell where `python` has access to all these libraries, isolated from your system Python.
