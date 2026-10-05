# Learning Log — GPU/PyTorch environment<Topic></topic>

## Date

2026-10-05

## Objective

Q: What was I trying to learn?

A: I tried to learn how cuda affects computations and speed up the matrix multiplication process.

## Why does this matter?

Q: Where is this used in modern ML/LLM systems?

A: This is the very basics of ML/LLM as each process is basically a mathematical calculation. So, learning how to do calculations using a framework on gpu's is the core of the building we are building in future.

## Resources studied

1. [pytorch.org](https://pytorch.org/)
   <resource>
2. [docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
   <resource>
3. [share.google/aimode/d9lieIs7NTpbhl9PR](https://share.google/aimode/d9lieIs7NTpbhl9PR)
   <paper>

## Concepts I learned

1. Tensors are basically like numpy arrays.
2. cuda helps computations faster by adding GPU processing.
3. using fp16 raw, will not help in reducing processing time if gpu wasn't warmed up before (just like a cold started engine performs lower than heated engine). so always 	`torch.cuda.synchronize()`	 when doing fp16 or lower matrix multiplication.  Reason is shared in Resource studied # 3.

## Mathematics / equations

```text
Matrix or Dot Multiplication
```
