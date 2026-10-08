# Learning Log — GPU and PyTorch Environment

## Date
2026-10-05

## Objective
Verify the PyTorch/CUDA environment and understand how tensor placement and
precision affect matrix-multiplication performance.

## Why does this matter?
Tensor operations are the computational foundation of machine learning and
LLM systems. Understanding devices, CUDA, and data types helps identify when
work is running on the CPU or GPU and how precision choices affect execution.

## Resources studied

1. [PyTorch](https://pytorch.org/)
2. [PyTorch Tensor Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
3. [Benchmarking discussion](https://share.google/aimode/d9lieIs7NTpbhl9PR)

## Concepts I learned

- A tensor is a multidimensional array used for numerical computation.
- CUDA allows PyTorch to execute supported operations on an NVIDIA GPU.
- Matrix multiplication requires the first matrix's column count to equal the
  second matrix's row count.
- CUDA operations are asynchronous; reliable GPU timing requires synchronization
  around the measured operation and a warm-up run.
- FP16 can be faster than FP32 for suitable operations on supported GPUs, but
  results depend on the workload and timing methodology.

## Mathematics / equations
```text
A: (5000, 3000)
B: (3000, 5000)
A @ B: (5000, 5000)
```

## Implementation
Created `system_check.py` to report the PyTorch version, CUDA availability,
GPU name, and VRAM. Created `benchmark_gpu.py` to allocate GPU tensors, run
matrix multiplication, compare CPU/GPU timings, and compare FP32/FP16 timings.

## Experiment

### Hypothesis
The RTX 3090 should execute a large matrix multiplication faster than the CPU,
and FP16 may improve GPU execution time compared with FP32.

### Setup
- CPU: Intel i5-12600F
- System memory: 64 GB DDR4
- GPU: NVIDIA RTX 3090, 24 GB VRAM, CUDA device 0
- PyTorch: 2.14.1+cu126
- Matrices: `(5000, 3000)` and `(3000, 5000)`

### Results
Recorded in the environment README:

| Measurement | Recorded time |
| ----------- | ------------: |
| CPU matrix multiplication | 0.339017 s |
| GPU matrix multiplication | 0.011384 s |
| GPU FP32 matrix multiplication | 0.011688 s |
| GPU FP16 matrix multiplication | 0.004662 s |

These are recorded results from one run, not a controlled benchmark. The
timing script should synchronize immediately before and after each measured
GPU operation and keep tensor printing outside the timed section for a more
reliable comparison.

## What broke?
CUDA's asynchronous execution makes naive wall-clock timings misleading unless
the device is synchronized around the measured operation.

## How I debugged it
Added warm-up matrix multiplications and `torch.cuda.synchronize()` calls
before timing FP32 and FP16 operations.

## What I changed
Added warm-up operations and CUDA synchronization to the precision comparison
in `benchmark_gpu.py`.

## What I learned from the failure
A warm-up helps avoid measuring initialization effects, and synchronization
ensures GPU work is complete before its runtime is reported. Synchronization
and output formatting should be kept outside the timed interval where possible.

## Final understanding
PyTorch tensors can be placed on a CUDA device to use GPU computation. GPU
matrix-multiplication performance can differ substantially from CPU
performance, and precision can affect runtime. Reliable comparisons require
controlled setup, warm-up, and correct synchronization.

## What I would improve next
Repeat the comparison over multiple runs, synchronize both before and after
each timed operation, and report a robust statistic such as the median.

## Evidence

- Code: `system_check.py` and `benchmark_gpu.py`
- Experiment: CPU/GPU and FP32/FP16 matrix multiplication
- Hardware: NVIDIA RTX 3090

## Status

- [x] Study
- [x] Implement
- [x] Experiment
- [x] Document
- [x] Publish
- [x] Complete
