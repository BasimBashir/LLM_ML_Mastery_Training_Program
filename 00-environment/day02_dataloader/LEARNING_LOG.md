# Learning Log — Dataset and DataLoader<Topic></topic>

## Date

2026-10-06

## Objective

Q: What was I trying to learn?

A: I was trying to learn about how a custom data can be loaded using pytorch.

## Why does this matter?

Q: Why does DataLoader exist?

A: Dataloader class exists to load dataset using pytorch and it supports vision, text and audio types. So, we dont have to build customloader ourselves.

Q: Why separate Dataset and DataLoader?

A: Dataset class helps to helps load dataset and dataloader makes it processable as we need.

Q: what happens when `num_workers > 0`?

A: When dataset is quite large then increasing the number of workers speeds up the data loading process. Just like multi-threaded operations in programming.

Q: What is a collate function?

A: this functions converts numpy arrays/python numerical values (basically raw data) into Tensors while preserving the data structure so GPU can load that data for processing. It can do individual sample level or on entire batch based on bach_size settings given by user.

Q: When can more workers make things worse?

A: More workers doesn't mean higher throughput. If data is small then increasing number of workers only produce overhead which causes the processing to actually slow down and especially when windows OS is involved on this.

Q: Explain the lifecycle of one batch.

A: I trained for 10 epochs and each batch caused training accuracy to jump up and training loss to go down as epoch1-10 (training accuracy: 90.65% -99.61%), (training loss: 0.2814 - 0.0091). So, training curve was good as accuracy was increasing and loss was decreasing but validation kept sinosoidal though non-linear as epoch1-10 (validation accuracy: 92.39% - 91.80%), (validation loss: 0.2336 - 0.6934). It seems validation data wasn't enough and generic but in general good model training.

Q: Document checkpoint contents.

A: Below are the full training ckpt contents:
Epoch 1/10 - train loss: 0.2814, train accuracy: 90.65% - validation loss: 0.2336, validation accuracy: 92.39%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 2/10 - train loss: 0.1407, train accuracy: 95.02% - validation loss: 0.2541, validation accuracy: 92.36%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 3/10 - train loss: 0.0690, train accuracy: 97.47% - validation loss: 0.3198, validation accuracy: 92.07%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 4/10 - train loss: 0.0342, train accuracy: 98.72% - validation loss: 0.4107, validation accuracy: 92.08%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 5/10 - train loss: 0.0200, train accuracy: 99.22% - validation loss: 0.5284, validation accuracy: 91.41%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 6/10 - train loss: 0.0163, train accuracy: 99.35% - validation loss: 0.5368, validation accuracy: 92.14%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 7/10 - train loss: 0.0132, train accuracy: 99.46% - validation loss: 0.5828, validation accuracy: 91.59%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 8/10 - train loss: 0.0102, train accuracy: 99.54% - validation loss: 0.6675, validation accuracy: 91.84%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 9/10 - train loss: 0.0096, train accuracy: 99.59% - validation loss: 0.6943, validation accuracy: 91.64%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt
Epoch 10/10 - train loss: 0.0091, train accuracy: 99.61% - validation loss: 0.6934, validation accuracy: 91.80%
Checkpoint saved to 00-environment\day02_dataloader\checkpoints\ag_news_mlp.pt

## Resources studied

1. [docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)<resource></resource>
2. [docs.pytorch.org/docs/2.14/data.html](https://docs.pytorch.org/docs/2.14/data.html)<resource></resource>
3. [chatgpt.com/share/6ac504ff-7abc-83ee-a379-6e256be7cd50](https://chatgpt.com/share/6ac504ff-7abc-83ee-a379-6e256be7cd50)
4. [docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

## Day 5 — AdamW, learning-rate scheduling and gradient accumulation

### Concepts

- **Adam** adapts each parameter's step using estimates of the first and second
  moments of its gradients.
- **AdamW** decouples weight decay from Adam's gradient-based update. The
  training script uses AdamW with configurable weight decay.
- The **learning rate** scales each optimizer update. Warmup increases it over
  the first configured fraction of optimizer steps; cosine decay then smoothly
  reduces it toward zero.
- **Gradient accumulation** adds gradients from several smaller forward/
  backward passes and applies one optimizer update at their boundary. Gradients
  are averaged by the number of samples in the group, including a short final
  group.

### Effective-batch experiment

Run `python 00-environment\day02_dataloader\benchmark.py` to train and compare
the two requested configurations with the same seed, AdamW settings, and
temporary checkpoints. The benchmark removes its temporary checkpoints when
both runs finish.

Both configurations have nominal effective batch size 8:

```text
8 examples/update = 8 x 1
8 examples/update = 2 x 4
```

For the same examples and fixed model parameters, averaging the four
batch-of-2 gradients gives the same mean gradient as a batch of 8, up to
floating-point summation differences. The two runs therefore make roughly the
same number of optimizer updates per epoch when the dataset size is divisible
by 8. They are not universally identical: stochastic/batch-dependent layers,
different example groupings, and a final incomplete group can change results.
This MLP has no BatchNorm or dropout, and the loop correctly normalizes a
partial final group by its actual sample count.

### Checkpoint/resume note

Checkpoints now include AdamW and scheduler state. The batch size,
accumulation count, warmup ratio, weight decay, and total schedule length must
match when resuming so that optimizer updates and the cosine schedule remain
consistent.

## Day 6 — Mixed precision

### Concepts

- **FP32** is the default full-precision mode and reference for comparing
  stability and validation loss.
- **FP16** has a smaller exponent range than FP32 and BF16, so values can
  underflow or overflow. CUDA FP16 runs under autocast with `GradScaler`: the
  loss is scaled before backpropagation, and gradients are unscaled before
  updates. The scaler skips an optimizer/scheduler update on non-finite
  gradients. FP16 is CUDA-only in this script.
- **BF16** has an exponent range similar to FP32 but fewer significand bits;
  it can represent a wide range of values with lower precision per value and
  generally does not need loss scaling. CUDA BF16 is rejected on unsupported
  devices.
- **Autocast** selects operation dtypes automatically while model parameters
  and optimizer state remain FP32. This improves compatibility and numerical
  stability compared with converting the entire model to half precision.

### Experiment

Run `python 00-environment\day02_dataloader\benchmark_amp.py` on a CUDA machine
to run the same seeded experiment in FP32, FP16, and BF16. It reports validation
loss, training throughput (samples/second), peak allocated and reserved VRAM,
and writes those measurements to
`00-environment\day02_dataloader\day6_amp_results.csv`. FP16 requires CUDA, so
the full three-mode comparison should be run on a CUDA GPU.

Measured on the RTX 3090 (`torch 2.14.1+cu126`), with batch size 64, one epoch,
and `--max-features 5000`:

| Mode | Peak VRAM allocated (MB) | Peak VRAM reserved (MB) | Speed (samples/s) | Validation loss |
| ---- | -----------------------: | ----------------------: | ----------------: | --------------: |
| FP32 | 44.06 | 62 | 2302.41 | 0.260619 |
| FP16 | 44.06 | 64 | 2124.27 | 0.260584 |
| BF16 | 44.06 | 64 | 2302.25 | 0.260779 |

On this small MLP and dense bag-of-words input, mixed precision did not reduce
measured peak allocated VRAM or improve throughput; FP16 was slower in this
single run. The validation losses were close. FP16 skipped five optimizer
updates after `GradScaler` detected non-finite gradients, demonstrating why
loss scaling and monitoring update counts matter. These measurements are
workload- and hardware-dependent, not a general mixed-precision performance
claim.

## Concepts I learned

1. Creating Custom Dataset
2. using Dataloader class
3. performing batch inspection
4. creatng custom collate function
5. Affect on processing with batch_size, num_workers, shuffle on/off
6. Training MLP from scratch without Trainer
7. learning to save in ckpts based on epochs

## Mathematics / equations

```text
None
```
