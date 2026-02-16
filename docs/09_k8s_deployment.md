# 09. Kubernetes Deployment: KubeRay

## Introduction
You run Ray on Kubernetes using the **KubeRay Operator**. This operator provides Custom Resource Definitions (CRDs) to manage Ray Clusters natively.

## The CRD: `RayService`

The `RayService` resource is the gold standard for production. It combines:
1.  **Infrastructure**: "I want a Ray Cluster with 3 GPU workers."
2.  **Application**: "I want to run `config.yaml` on it."

## The Manifest: `ray-service.yaml` (Excerpt)

```yaml
kind: RayService
spec:
  # THE APPLICATION
  serveConfigV2: |
    applications:
      - name: my-app
        import_path: production_graph:app
        deployments:
          - name: BatchModel
            num_replicas: 2
            ray_actor_options:
              num_gpus: 1

  # THE INFRASTRUCTURE
  rayClusterConfig:
    rayVersion: '2.9.0'
    headGroupSpec:
      # ... config for head pod ...
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 3
        template:
           # ... standard PodSpec ...
           resources:
             limits:
               nvidia.com/gpu: 1
```

## The Zero-Downtime Workflow
1.  **Update**: You change the `serveConfigV2` (e.g., bump replicas from 2 to 5).
2.  **Apply**: `kubectl apply -f ray-service.yaml`.
3.  **Reconcile**: The Operator sees the change.
    *   It brings up the new configuration.
    *   It waits for the new replicas to be healthy.
    *   It switches the traffic.
    *   It drains the old replicas.

## GitOps Workflow
This facilitates a GitOps workflow. You define the desired state in Git, and the KubeRay Operator runs a reconciliation loop to bring the cluster to that state.
