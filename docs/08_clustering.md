# 08. Clustering: Head & Workers

## Introduction
So far, `ray.init()` started everything on your laptop. In production, Ray runs as a **Cluster**.
A Ray Cluster is remarkably similar to a Kubernetes Cluster in architecture.

## The Architecture

### 1. The Head Node
*   **Role**: The Control Plane.
*   **Components**:
    *   **GCS (Global Control Service)**: The metadata store. Backed by Redis. Stores the cluster table, actor information, and object location table.
    *   **Dashboard**: The Web UI for metrics and log visualization.
    *   **Driver**: The script that submits work.

### 2. The Worker Nodes
*   **Role**: Compute.
*   **Components**:
    *   **Raylet**: The local scheduler and node manager. It handles resource management (Leasing CPUs/GPUs) and task execution.
    *   **Object Store (Plasma)**: A shared memory segment (tmpfs/shm) for storing immutable objects (tensors, dataframes). It allows zero-copy reads by multiple workers on the same node.

## The Simulation: `manual_cluster.sh`

```bash
# 1. Start the Head
# --head: Tells this node to run GCS.
# --include-dashboard=true: Enables the Web UI.
# --dashboard-host=0.0.0.0: Allows external access (Crucial in containers/Nix).
ray start --head --port=6379 --include-dashboard=true --dashboard-host=0.0.0.0

# 2. Start a Worker (on the same machine for demo)
# --address: Tells this node where the GCS is.
ray start --address=localhost:6379
```

## Resource Scheduling
When you define `@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 1})`:
1.  Ray Serve Controller asks GCS: "Where can I fit 1 CPU and 1 GPU?"
2.  GCS checks the cluster state.
3.  If Node A has a GPU free, the Actor is scheduled there.

This **Resource Logical View** abstracts away the physical machines. You just ask for resources, and Ray places the actors.
