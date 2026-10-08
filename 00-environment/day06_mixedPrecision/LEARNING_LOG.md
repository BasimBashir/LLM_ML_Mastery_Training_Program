# Learning Log — Mixed Precision

## Date
2026-10-06

## Objective
Compare FP32, FP16, and BF16 training for validation loss, throughput, and
peak GPU memory, and understand the role of autocast and loss scaling.

## Why does this matter?
Mixed precision can reduce memory use and improve throughput on supported
hardware, but numerical stability and actual performance depend on the
workload and device.

## Resources studied

1. [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
2. [PyTorch AMP Recipe](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

## Concepts I learned

- FP32 is the full-precision reference for stability and validation loss.
- FP16 has a smaller exponent range than FP32 and BF16. `GradScaler` scales
  the loss and skips updates when gradients are non-finite.
- BF16 has a wide exponent range similar to FP32, but fewer significand bits;
  it generally does not require loss scaling.
- Autocast selects operation dtypes while model parameters and optimizer state
  remain FP32.
- Mixed precision does not necessarily improve speed or memory usage for every
  workload.

## Mathematics / equations
```text
None
```

## Implementation
`benchmark_amp.py` runs the same seeded training experiment in FP32, FP16, and
BF16, reporting validation loss, samples/second, and peak allocated/reserved
VRAM. Results are written to `day6_amp_results.csv`.

## Experiment

### Hypothesis
FP16 and BF16 may reduce memory use or improve throughput relative to FP32,
with validation loss remaining close.

### Setup
- Model: AG News MLP with dense bag-of-words input
- GPU: NVIDIA RTX 3090
- PyTorch: 2.14.1+cu126
- Batch size: 64
- Epochs: 1
- Maximum features: 5,000

### Results

| Mode | Peak VRAM allocated (MB) | Peak VRAM reserved (MB) | Speed (samples/s) | Validation loss |
| ---- | -----------------------: | ----------------------: | ----------------: | --------------: |
| FP32 | 44.06 | 62 | 2302.41 | 0.260619 |
| FP16 | 44.06 | 64 | 2124.27 | 0.260584 |
| BF16 | 44.06 | 64 | 2302.25 | 0.260779 |

On this small MLP, mixed precision did not reduce measured allocated VRAM or
improve throughput; FP16 was slower in this run. Validation losses were close.
FP16 skipped five optimizer updates after `GradScaler` detected non-finite
gradients. These measurements describe this workload and run, not a general
performance result.

## What broke?
FP16 encountered non-finite gradients and skipped five optimizer updates.
Mixed precision also failed to improve speed or measured allocated VRAM in
this small workload.

## How I debugged it
Compared validation loss, samples per second, peak allocated/reserved memory,
and optimizer update behavior across precision modes.

## What I changed
No specific change to the experiment was recorded.

## What I learned from the failure
Loss scaling can help FP16 training handle limited dynamic range, but skipped
updates should be monitored. Precision mode should be selected from measured
behavior on the intended workload rather than assumed performance benefits.

## Final understanding
Autocast enables lower-precision operations while keeping model parameters and
optimizer state in FP32. FP16 uses gradient scaling to manage numerical range;
BF16 has a wider exponent range. The speed and memory benefits must be measured
for the model and hardware in use.

## What I would improve next
Repeat the measurements over multiple runs and compare larger models or
sequence-based workloads where activation memory and compute are more
substantial.

## Evidence

- Code: `benchmark_amp.py` and `../day04_TrainingLoop/train.py`
- Experiment: `day6_amp_results.csv`
- GPU: NVIDIA RTX 3090

## Status

- [x] Study
- [x] Implement
- [x] Experiment
- [x] Document
- [ ] Publish
- [ ] Complete
