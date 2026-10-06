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

## Resources studied

1. [docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
   <resource>
2. [docs.pytorch.org/docs/2.14/data.html](https://docs.pytorch.org/docs/2.14/data.html)
   <resource>
3. [chatgpt.com/share/6ac504ff-7abc-83ee-a379-6e256be7cd50](https://chatgpt.com/share/6ac504ff-7abc-83ee-a379-6e256be7cd50)
   <paper>

## Concepts I learned

1. Creating Custom Dataset
2. using Dataloader class
3. performing batch inspection
4. creatng custom collate function
5. Affect on processing with batch_size, num_workers, shuffle on/off

## Mathematics / equations

```text
None
```
