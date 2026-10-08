# Learning Log — Training Loop

## Date
2026-10-06

## Objective
Implement a complete PyTorch training loop with validation, checkpoint saving,
and resume support.

## Why does this matter?
Training loops are the core control flow for model optimization. Explicitly
handling forward passes, loss, gradients, optimizer updates, validation, and
checkpoints makes training behavior inspectable and recoverable.

## Resources studied

1. [PyTorch Optimization Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
2. [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
3. [PyTorch Dataset and DataLoader Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

## Concepts I learned

- A training batch runs through forward pass, loss calculation, backpropagation,
  and an optimizer update.
- Gradients must be cleared between optimizer updates.
- Validation runs without gradient updates and tracks generalization.
- Checkpoints need model and optimizer state to resume training meaningfully.
- Training accuracy can improve while validation loss worsens, indicating
  overfitting.

## Mathematics / equations
```text
None
```

## Implementation
Implemented the AG News MLP training and validation loops in `train.py`.
Checkpoints contain model, optimizer, scheduler, and scaler states; epoch and
global-step progress; vocabulary and training configuration; and CPU/GPU
random-number generator state. Checkpoint files are saved atomically and can
be loaded to resume training.

## Experiment

### Hypothesis
A training loop that saves complete state after each epoch should allow a run
to resume without losing optimizer history or schedule progress.

### Setup
- Model: two-hidden-layer MLP
- Dataset: AG News
- Epochs: 10
- Checkpoint: `checkpoints/ag_news_mlp.pt`

### Results

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
| ----: | ---------: | -------------: | --------------: | ------------------: |
| 1 | 0.2814 | 90.65% | 0.2336 | 92.39% |
| 2 | 0.1407 | 95.02% | 0.2541 | 92.36% |
| 3 | 0.0690 | 97.47% | 0.3198 | 92.07% |
| 4 | 0.0342 | 98.72% | 0.4107 | 92.08% |
| 5 | 0.0200 | 99.22% | 0.5284 | 91.41% |
| 6 | 0.0163 | 99.35% | 0.5368 | 92.14% |
| 7 | 0.0132 | 99.46% | 0.5828 | 91.59% |
| 8 | 0.0102 | 99.54% | 0.6675 | 91.84% |
| 9 | 0.0096 | 99.59% | 0.6943 | 91.64% |
| 10 | 0.0091 | 99.61% | 0.6934 | 91.80% |

The training metrics improved steadily while validation loss rose after the
first epoch. This is evidence of overfitting and suggests that generalization
needs further work.

## What broke?
Validation loss increased as training loss continued to fall; training accuracy
alone did not reflect generalization.

## How I debugged it
Compared training and validation loss and accuracy across all ten epochs.

## What I changed
No specific change to address the validation trend was recorded.

## What I learned from the failure
Improving training metrics does not ensure better validation performance.
Checkpointing preserves the run state but does not prevent overfitting.

## Final understanding
Each batch produces predictions and a loss; gradients from that loss update
the model. Validation measures performance without changing weights, and a
complete checkpoint preserves enough state to continue the same training run.

## What I would improve next
Investigate regularization and other ways to reduce the widening training/
validation gap, then test checkpoint resume after interrupting a run.

## Evidence

- Code: `train.py`
- Checkpoint: `checkpoints/ag_news_mlp.pt`
- Experiment: 10-epoch AG News training and validation metrics

## Status

- [x] Study
- [x] Implement
- [x] Experiment
- [x] Document
- [ ] Publish
- [ ] Complete
