# Learning Log — AdamW, Learning-Rate Scheduling, and Gradient Accumulation

## Date
2026-10-06

## Objective
Use AdamW with warmup and cosine learning-rate decay, and compare a larger
batch with gradient accumulation over smaller batches.

## Why does this matter?
Optimizer and schedule choices determine how model parameters change during
training. Gradient accumulation makes it possible to approximate a larger
effective batch when memory limits prevent processing that batch at once.

## Resources studied

1. [PyTorch AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
2. [Hugging Face optimization concepts](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules)

## Concepts I learned

- Adam adapts parameter updates using first- and second-moment estimates of
  gradients.
- AdamW decouples weight decay from Adam's gradient-based update.
- Warmup increases the learning rate over the configured initial steps; cosine
  decay then reduces it smoothly.
- Gradient accumulation sums gradients across micro-batches and applies one
  optimizer update at the accumulation boundary.
- The final partial accumulation group must be normalized by its actual sample
  count.
- Resuming requires matching settings that determine optimizer updates and
  schedule length.

## Mathematics / equations
```text
effective batch size = micro-batch size x accumulation steps
8 = 8 x 1 = 2 x 4
```

## Implementation
The Day 4 training loop supports AdamW, configurable weight decay, warmup,
cosine scheduling, and gradient accumulation. `benchmark.py` runs both
effective-batch configurations with the same seed and temporary checkpoints.

## Experiment

### Hypothesis
For fixed model parameters and the same examples, averaging gradients from
four micro-batches of size 2 should closely match the mean gradient from a
batch of 8, apart from floating-point summation differences.

### Setup
- Model: AG News MLP without BatchNorm or dropout
- Configuration A: batch size 8, accumulation steps 1
- Configuration B: batch size 2, accumulation steps 4
- Seed and AdamW settings: held constant
- Checkpoints: temporary files, removed after both runs finish

### Results
The script reports runtime for both configurations and their runtime ratio.
Numeric run results were not retained in this learning log. The two settings
have nominal effective batch size 8, but results can differ with different
example groupings, a final incomplete group, stochastic layers, or
batch-dependent layers.

## What broke?
No specific optimizer or accumulation failure was recorded.

## How I debugged it
No debugging steps were recorded.

## What I changed
No corrective change was recorded.

## What I learned from the failure
No failure was recorded. Effective batch size alone does not guarantee
identical updates in every model or dataset.

## Final understanding
Gradient accumulation reduces micro-batch memory requirements while retaining
approximately the same effective batch size. Correct sample-based gradient
normalization and a schedule aligned to optimizer updates are important,
particularly for an incomplete final group and checkpoint resume.

## What I would improve next
Run the comparison and save both runtime measurements and resulting validation
metrics alongside the experiment configuration.

## Evidence

- Code: `../day04_TrainingLoop/train.py`
- Experiment: `benchmark.py`

## Status

- [x] Study
- [x] Implement
- [ ] Experiment
- [x] Document
- [ ] Publish
- [ ] Complete
