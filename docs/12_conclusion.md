# 12. Conclusion & Further Reading

## Series Recap
Congratulations! You have built a Production-Grade ML Serving platform from scratch on NixOS.

We have bridged the gap between **ML Concepts** and **Systems Engineering**:
1.  **Actors** are just **Stateful Processes**.
2.  **Tensors** are just **Shared Memory Buffers**.
3.  **Model Training** is just **Iterative State Mutation**.
4.  **Clustering** is just **RPC + Resource Scheduling**.

## Where to go from here?

Ray Serve is the foundation for the world's largest AI applications (ChatGPT, Uber, Spotify). Here are advanced topics to explore next:

### 1. Large Language Models (LLMs)
Serving LLMs (like Llama-3 or Mistral) requires specialized techniques because the models often don't fit on a single GPU.
*   **Technique**: Model Parallelism (Sharding a model across GPUs).
*   **Ray Library**: [Ray Serve Tutorials](https://docs.ray.io/en/latest/serve/tutorials/index.html) (Check for LLM/vLLM examples).
*   **Use Case**: Creating a private, self-hosted version of OpenAI API.

### 2. Generative AI (Stable Diffusion)
Scaling image generation requires massive throughput management.
*   **Technique**: Autoscaling based on Queue Depth.
*   **Ray Feature**: [Fractional GPUs](https://docs.ray.io/en/latest/serve/scaling-and-resource-allocation.html#resource-management-cpus-gpus-and-custom-resources) (Packing 4 models onto 1 A100).

### 3. Distributed Fine-Tuning
Training models on your own data.
*   **Library**: [Ray Train](https://docs.ray.io/en/latest/train/train.html).
*   **Pattern**: Use Ray Train to update weights, then hot-swap the new weights into your running Ray Serve actors without downtime.

## Final Thoughts
As a Software Engineer, your job isn't to design the Neural Network layers. Your job is to build the **Engine** that runs them reliably, cost-effectively, and at scale.

Ray provides the primitives. NixOS provides the reproducibility. You provide the architecture.

**Happy Building.**
