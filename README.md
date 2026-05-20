# Chapel GPU Demo

This is the Chapel code for a demo of GPU programming in Chapel, presented at Northwest C++ Users' Group in May 2026.

This code is available at https://github.com/jabraham17/northwestcpp-demo.

## Demos

### 01 — GPU Sigmoid
Implements a sigmoid function offloaded to a single GPU. Introduces Chapel GPU programming basics: `on here.gpus[0]` to move computation to a GPU sublocale, explicit CPU-to-GPU data copies, and GPU kernel execution.

### 02 — Multi-GPU Sigmoid
Extends the sigmoid to run across multiple GPUs on a single node. Uses `RangeChunk` to partition data and `coforall` to launch concurrent GPU kernels, demonstrating how to distribute work across all available GPUs.

### 03 — Distributed GPU Sigmoid
Scales the multi-GPU sigmoid across multiple nodes using `BlockDist`. Combines distributed arrays with per-node multi-GPU execution, showing how Chapel composes distribution and GPU offloading.

### 04 — Heterogeneous Sigmoid
Splits computation between CPUs and GPUs on each node using `cobegin`. A configurable `gpuPercent` parameter controls the CPU/GPU work split, demonstrating heterogeneous execution across a distributed system.

## Building and Running

You can download Chapel and Mason from the [Chapel website](https://chapel-lang.org/download/). Follow the installation instructions to set up your Chapel environment.

After setting up your Chapel environment, navigate into a demo directory and use `mason` to build and run.

### GPU Support

These demos require Chapel to be built with GPU support:

```bash
export CHPL_LOCALE_MODEL=gpu
export CHPL_GPU=amd           # or nvidia
export CHPL_GPU_ARCH=gfx942   # set to your target architecture
# ... rebuild Chapel with GPU support ...
cd 01_GpuSigmoid
mason build
mason run
```

### Multi-Locale (distributed memory)

Demos 03 and 04 require Chapel to be built with a communication layer for multi-locale execution:

```bash
export CHPL_COMM=ofi   # or gasnet
# ... build Chapel with the chosen communication layer ...
cd 03_DistributedGpuSigmoid
mason build
mason run -- --numLocales=4
```

### Graphing Output

Use the `--printForGraph` flag to output data points, then pipe to `graph.py` to visualize:

```bash
mason run -- --printForGraph | python3 ../graph.py
```
