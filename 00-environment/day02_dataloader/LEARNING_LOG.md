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
4. [docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)<paper></paper>

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
