# Learning Log — Dataset and DataLoader

## Date
2026-10-06

## Objective
Learn how to load and batch a custom text dataset with PyTorch's `Dataset` and
`DataLoader` APIs.

## Why does this matter?
A `Dataset` defines how individual samples are retrieved, while a `DataLoader`
handles batching, shuffling, worker processes, and collation. Keeping these
responsibilities separate makes data input reusable across training loops and
supports text, vision, and audio workloads.

More workers can improve throughput when sample loading is expensive, but they
also introduce process startup and coordination overhead. For small datasets,
and especially on Windows, additional workers can make loading slower.

A collate function combines individual samples into a batch. This dataset's
custom collate function returns text and label lists while preserving their
batch structure.

## Resources studied

1. [PyTorch Dataset and DataLoader Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
2. [PyTorch Data API](https://docs.pytorch.org/docs/2.14/data.html)
3. [Build the Neural Network Model](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
4. [Study discussion](https://chatgpt.com/share/6ac504ff-7abc-83ee-a379-6e256be7cd50)

## Concepts I learned

- A custom `Dataset` implements `__len__` and `__getitem__`.
- `DataLoader` batches samples and can shuffle them or load them with workers.
- A custom collate function converts a list of samples into the desired batch
  representation.
- Larger `num_workers` values do not guarantee higher throughput.
- Batch size, worker count, and shuffling should be benchmarked for the workload.

## Mathematics / equations
```text
None
```

## Implementation
Built `AGNewsDataset`, `custom_collate_fn`, `load_dataset`, batch inspection,
and a DataLoader benchmark in `dataset.py`.

## Experiment

### Hypothesis
Batch size, worker count, and shuffling can change input pipeline throughput;
the best settings depend on dataset and platform overhead.

### Setup
- Dataset: AG News
- Batch sizes: 1, 8, 32
- Worker counts: 0, 1, 2, 4
- Shuffle: enabled and disabled

### Results
The code measures samples/second and batches/second for each configuration.
Numeric throughput results were not retained in this learning log.

## What broke?
No specific DataLoader failure was recorded.

## How I debugged it
No debugging steps were recorded.

## What I changed
No corrective change was recorded.

## What I learned from the failure
No failure was recorded.

## Final understanding
The dataset defines the sample interface; the DataLoader turns those samples
into batches and controls how they are delivered. Worker count and batch size
are workload-dependent settings, not universal speed switches.

## What I would improve next
Record the measured throughput for each benchmark configuration and compare
the results across worker counts.

## Evidence

- Code: `dataset.py`
- Experiment: DataLoader benchmark in `dataset.py`
- Dataset: AG News

## Status

- [x] Study
- [x] Implement
- [x] Experiment
- [x] Document
- [ ] Publish
- [ ] Complete
