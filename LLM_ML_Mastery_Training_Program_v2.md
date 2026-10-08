# LLM & Machine Learning Mastery — Study → Build → Document → Publish

> **Start date:** 2026-10-05
> **Target pace:** 15–20 hours/week
> **Estimated duration:** 9–12 months
> **Hardware:** RTX 3090 24 GB VRAM · Intel i5-12600F · 64 GB RAM
> **Primary objective:** Become capable of understanding, implementing, training, evaluating, optimizing and deploying modern ML/LLM systems — not merely using LLM frameworks.

---

# 1. How to use this program

This version is deliberately different from a normal course list.

**Never treat a checkbox as complete merely because you watched a lecture.**

Every topic follows this pipeline:

```text
1. STUDY
      ↓
2. UNDERSTAND
      ↓
3. IMPLEMENT
      ↓
4. EXPERIMENT
      ↓
5. DOCUMENT
      ↓
6. PUBLISH
      ↓
7. CHECK ✓
```

For example:

```text
Topic: PyTorch DataLoader

STUDY
→ read PyTorch Dataset/DataLoader documentation

UNDERSTAND
→ know Dataset vs DataLoader vs sampler vs batch

IMPLEMENT
→ write your own dataset + DataLoader

EXPERIMENT
→ compare batch sizes / num_workers

DOCUMENT
→ explain what you learned and what broke

PUBLISH
→ commit code + learning note to GitHub

CHECK
→ mark the task complete
```

This is the operating system for the entire curriculum.

---

# 2. Your portfolio strategy

Do **not** make one giant README saying:

> "I learned LLMs."

Instead, build a public engineering trail.

Your GitHub should show:

```text
basim-llm-journey/
│
├── README.md
│
├── 00-environment/
├── 01-ml-foundations/
├── 02-nlp-foundations/
├── 03-attention/
├── 04-transformer-from-scratch/
├── 05-tokenization/
├── 06-pretraining/
├── 07-modern-transformers/
├── 08-sft/
├── 09-lora-qlora/
├── 10-evaluation/
├── 11-preference-learning/
├── 12-rlhf/
├── 13-grpo-rlvr/
├── 14-distillation/
├── 15-inference/
├── 16-distributed-training/
├── 17-cuda-triton/
├── 18-rag/
├── 19-agents/
├── 20-multimodal/
├── 21-interpretability/
├── 22-safety-alignment/
└── 23-capstone/
```

Every module should contain:

```text
README.md
LEARNING_LOG.md
experiments/
src/
tests/
configs/
results/
figures/
```

For larger projects:

```text
project/
├── README.md
├── LEARNING_LOG.md
├── CHANGELOG.md
├── requirements.txt / pyproject.toml
├── configs/
├── data/
│   └── README.md
├── src/
├── scripts/
├── tests/
├── notebooks/
├── experiments/
├── results/
├── figures/
└── checkpoints/
    └── README.md
```

**Never commit large datasets or model checkpoints to GitHub.**

Use:

- [Git LFS](https://git-lfs.com/)
- [Hugging Face Hub](https://huggingface.co/)
- [Weights &amp; Biases](https://wandb.ai/)
- [MLflow](https://mlflow.org/)

---

# 3. The documentation template you will use for EVERY module

Create `LEARNING_LOG.md` in every module.

Copy this template:

```markdown
# Learning Log — <Topic>

## Date
YYYY-MM-DD

## Objective
What was I trying to learn?

## Why does this matter?
Where is this used in modern ML/LLM systems?

## Resources studied

1. <resource>
2. <resource>
3. <paper>

## Concepts I learned

- 
- 
- 

## Mathematics / equations

```text
put important equations here
```

## Implementation

What did I build?

## Experiment

### Hypothesis

What did I expect?

### Setup

- Model:
- Dataset:
- GPU:
- Batch:
- Sequence length:
- Learning rate:
- Precision:

### Results

| Experiment   | Metric | Result |
| ------------ | -----: | -----: |
| baseline     |        |        |
| experiment A |        |        |
| experiment B |        |        |

## What broke?

## How I debugged it

## What I changed

## What I learned from the failure

## Final understanding

Explain the concept in your own words without copying the source.

## What I would improve next

## Evidence

- Code:
- Experiment:
- Figure:
- Model:
- Dataset:

## Status

- [ ] Study
- [ ] Implement
- [ ] Experiment
- [ ] Document
- [ ] Publish
- [ ] Complete

```

This becomes your **learning evidence**.

---

# 4. Weekly public progress format

At the end of every week create:

```text
weekly/YYYY-WXX.md
```

Use:

```markdown
# Week XX — <Topic>

## What I learned

- 
- 
- 

## What I implemented

- 
- 
- 

## Biggest debugging problem

### Problem

### Root cause

### Fix

### Lesson

## Experiment

### Hypothesis

### Result

## What surprised me?

-

## Next week

- 

## Evidence

GitHub:
Hugging Face:
Experiment dashboard:
```

This is what you can later convert into LinkedIn posts.

---

# 5. Git commit convention

Use commits that explain the learning progression.

```text
feat: implement causal self attention
feat: add rotary positional embeddings
experiment: compare batch sizes
experiment: compare bf16 and fp16
fix: correct causal attention mask
fix: resume checkpoint optimizer state
docs: explain AdamW and weight decay
docs: document pretraining experiment
benchmark: measure tokens per second
refactor: separate tokenizer and dataset pipeline
```

Avoid:

```text
update
changes
final
test
stuff
```

Your Git history should itself communicate engineering maturity.

---

# 6. LinkedIn strategy

Do **not** post every tiny task.

Post after meaningful milestones.

Recommended milestones:

1. PyTorch/LLM environment
2. Attention from scratch
3. GPT from scratch
4. Custom tokenizer
5. First pretrained model
6. Data pipeline
7. SFT
8. LoRA/QLoRA comparison
9. DPO
10. GRPO/RLVR
11. Evaluation framework
12. Quantized inference server
13. Distributed training
14. CUDA/Triton optimization
15. RAG evaluation
16. Agent system
17. Multimodal model
18. Final foundation-model capstone

A strong post format:

```text
I spent the last X weeks studying <topic>.

What I learned:
• ...
• ...
• ...

What I built:
• ...

One problem I encountered:
...

Root cause:
...

Fix:
...

Most interesting result:
...

Code / experiment:
<GitHub link>

Next:
...
```

The emphasis should be **engineering evidence**, not motivational content.

---

# 7. Phase 0 — Environment

# Week 1 — Build your ML laboratory

## Day 1 — GPU/PyTorch environment

### STUDY

Read first:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [PyTorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

Understand:

- [X] Tensor
- [X] device
- [X] CPU vs GPU
- [X] CUDA
- [X] tensor dtype
- [X] FP32
- [X] FP16
- [X] BF16

### IMPLEMENT

Create:

```text
00-environment/
├── README.md
├── system_check.py
├── benchmark_gpu.py
└── LEARNING_LOG.md
```

Implement:

- [X] Check PyTorch version.
- [X] Check CUDA availability.
- [X] Print GPU name.
- [X] Print VRAM.
- [X] Allocate tensor on GPU.
- [X] Run matrix multiplication.
- [X] Measure execution time.

### EXPERIMENT

- [X] Compare CPU vs GPU matrix multiplication.
- [X] Compare FP32 vs FP16.
- [X] Compare different matrix sizes.

### DOCUMENT

- [X] Record exact hardware.
- [X] Record software versions.
- [X] Record benchmark results.
- [X] Explain why GPU wins.

### PUBLISH

- [X] Commit code.
- [X] Write `LEARNING_LOG.md`.
- [X] Push to GitHub.

### COMPLETE

- [X] Study
- [X] Implement
- [X] Experiment
- [X] Document
- [X] Publish
- [X] Complete

---

# Day 2–3 — Dataset and DataLoader

## STUDY FIRST

Do not code immediately.

Read:

- [PyTorch Dataset and DataLoader Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Dataset API](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
- [DataLoader API](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

Understand:

- [X] Dataset
- [X] `__len__`
- [X] `__getitem__`
- [X] DataLoader
- [X] batching
- [X] shuffling
- [X] workers
- [X] collate function
- [X] sampler

## IMPLEMENT

Create:

```text
00-environment/day02_dataloader/
├── dataset.py
└── LEARNING_LOG.md
```

Build:

- [X] Custom Dataset.
- [X] DataLoader.
- [X] Batch inspection.
- [X] Custom collate function.
- [X] Configurable batch size.
- [X] Benchmark batch size, worker count, and shuffle settings.

## EXPERIMENT

Test:

- [X] batch size 1
- [X] batch size 8
- [X] batch size 32
- [X] different `num_workers`
- [X] shuffle on/off

Record throughput.

## DOCUMENT

Answer:

1. Why does DataLoader exist?
2. Why separate Dataset and DataLoader?
3. What happens when `num_workers > 0`?
4. What is a collate function?
5. When can more workers make things worse?

- [X] Write answers in `LEARNING_LOG.md`.

---

# Day 4 — Training loop

## STUDY

Read:

- [PyTorch Optimization Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

Learn:

- [X] forward pass
- [X] loss
- [X] backward pass
- [X] gradients
- [X] optimizer
- [X] `zero_grad`
- [X] `step`

## IMPLEMENT

- [X] Write a training loop without Trainer.
- [X] Add validation.
- [X] Add checkpoint saving.
- [X] Add checkpoint loading.

## EXPERIMENT

- [X] Train an MLP.
- [X] Save checkpoint.
- [X] Kill process.
- [X] Resume training.
- [X] Verify optimizer state also resumes.

## DOCUMENT

- [X] Explain the lifecycle of one batch.
- [X] Document checkpoint contents.

Keep the training implementation and its checkpoints in:

```text
00-environment/day04_TrainingLoop/
├── train.py
└── checkpoints/
```

---

# Day 5 — AdamW, LR scheduling and gradient accumulation

## STUDY

- [PyTorch AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [Hugging Face optimization concepts](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules)

Learn:

- [X] Adam
- [X] AdamW
- [X] learning rate
- [X] warmup
- [X] cosine decay
- [X] gradient accumulation

## IMPLEMENT

- [X] AdamW.
- [X] warmup.
- [X] cosine scheduler.
- [X] gradient accumulation.

## EXPERIMENT

Compare:

```text
batch=8
gradient_accumulation=1

vs

batch=2
gradient_accumulation=4
```

Explain why the effective batch size is approximately equivalent.

Keep the comparison script in:

```text
00-environment/day05_modelOptimization-AdamW/benchmark.py
```

---

# Day 6 — Mixed precision

## STUDY

- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [PyTorch AMP Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

Learn:

- [X] FP32
- [X] FP16
- [X] BF16
- [X] autocast
- [X] GradScaler
- [X] numerical stability

## IMPLEMENT

- [X] FP32 training.
- [X] mixed-precision training.
- [X] record VRAM and speed.

## EXPERIMENT

Compare:

| Mode | VRAM | Speed | Loss |
| ---- | ---: | ----: | ---: |
| FP32 |      |       |      |
| FP16 |      |       |      |
| BF16 |      |       |      |

Keep the comparison script and results in:

```text
00-environment/day06_mixedPrecision/
├── benchmark_amp.py
└── day6_amp_results.csv
```

---

# Day 7 — Week 1 review

- [ ] Rebuild the training loop from memory.
- [ ] Explain every tensor shape.
- [ ] Review all failed experiments.
- [ ] Write `weekly/2026-W41.md`.
- [ ] Push GitHub changes.
- [ ] Create your first public progress post only if the milestone is substantial.

---

# PHASE 1 — NLP foundations

# Week 2 — Language modeling before Transformers

## Study order

### Step 1 — Read

- [Stanford Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/)
- [Dive into Deep Learning NLP](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html)

### Step 2 — Learn

- [ ] language modeling
- [ ] token prediction
- [ ] embeddings
- [ ] Word2Vec
- [ ] CBOW
- [ ] Skip-gram
- [ ] negative sampling
- [ ] RNN
- [ ] LSTM
- [ ] GRU
- [ ] Seq2Seq
- [ ] teacher forcing

### Step 3 — Implement

- [ ] Bigram language model.
- [ ] Character RNN.
- [ ] Tiny LSTM language model.

### Step 4 — Experiment

Compare:

```text
Bigram
RNN
LSTM
```

Record:

- [ ] validation loss
- [ ] training speed
- [ ] generated text quality
- [ ] context limitations

### Step 5 — Document

Explain:

> Why did the field move from RNNs toward attention?

### Step 6 — Publish

- [ ] Code
- [ ] plots
- [ ] learning log
- [ ] weekly report

---

# PHASE 2 — Attention

# Week 3 — Attention from first principles

## STUDY

Read/watch in this order:

1. [Karpathy — Let&#39;s build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
2. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Learn:

- [ ] Query
- [ ] Key
- [ ] Value
- [ ] dot product
- [ ] scaling
- [ ] softmax
- [ ] masking
- [ ] causal masking
- [ ] self-attention
- [ ] multi-head attention

## IMPLEMENT

Create:

```text
03-attention/
├── attention.py
├── visualize.py
├── tests/
└── LEARNING_LOG.md
```

Implement attention without importing a ready-made attention layer.

## EXPERIMENT

- [ ] Change number of heads.
- [ ] Change sequence length.
- [ ] Visualize attention.
- [ ] Compare masked vs unmasked attention.

## DOCUMENT

Draw the tensor shapes:

```text
X
↓
Q K V
↓
QKᵀ
↓
scale
↓
mask
↓
softmax
↓
weighted V
```

---

# Week 4 — Transformer from scratch

## STUDY

- [ ] Rewatch Karpathy GPT implementation.
- [ ] Read [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
- [ ] Read the original Transformer paper.

Learn:

- [ ] embeddings
- [ ] positional information
- [ ] residual connections
- [ ] LayerNorm
- [ ] MLP
- [ ] Transformer block
- [ ] LM head
- [ ] autoregressive generation

## IMPLEMENT — MAJOR PROJECT #1

Build:

```text
04-transformer-from-scratch/
├── model.py
├── tokenizer.py
├── train.py
├── generate.py
├── config.py
├── tests/
├── experiments/
├── figures/
├── README.md
└── LEARNING_LOG.md
```

Do not use Hugging Face `AutoModelForCausalLM` for the core model.

## EXPERIMENT

- [ ] Model size A.
- [ ] Model size B.
- [ ] Different context lengths.
- [ ] Different learning rates.
- [ ] temperature sampling.
- [ ] top-k.
- [ ] top-p.

## DOCUMENT

- [ ] Parameter count.
- [ ] FLOP estimate.
- [ ] VRAM usage.
- [ ] tokens/sec.
- [ ] loss curves.
- [ ] generated samples.

## PUBLISH

- [ ] GitHub README.
- [ ] Architecture diagram.
- [ ] Training graph.
- [ ] Results table.
- [ ] Learning log.
- [ ] GitHub release/tag.

---

# PHASE 3 — Tokenization

# Week 5 — BPE and modern tokenizers

## STUDY

In this order:

1. [Hugging Face Tokenizers Course](https://huggingface.co/learn/llm-course/chapter6/1)
2. [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
3. [SentencePiece](https://github.com/google/sentencepiece)
4. [Karpathy — Let&#39;s build the GPT tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

Learn:

- [ ] character tokenizer
- [ ] BPE
- [ ] WordPiece
- [ ] Unigram
- [ ] vocabulary
- [ ] special tokens
- [ ] normalization
- [ ] pre-tokenization
- [ ] encode/decode

## IMPLEMENT

- [ ] Implement minimal BPE yourself.
- [ ] Train tokenizer.
- [ ] Save vocabulary.
- [ ] Encode/decode text.

## EXPERIMENT

Compare token efficiency for:

- [ ] English
- [ ] Urdu
- [ ] Arabic
- [ ] Python
- [ ] JSON
- [ ] mixed technical text

## DOCUMENT

Explain:

> Why can tokenizer design materially affect multilingual model efficiency?

---

# PHASE 4 — Pretraining

# Weeks 6–10

Your main course from here:

## PRIMARY STUDY RESOURCE

[Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/)

Use its lectures/notes/assignments as the backbone.

Supplement with:

- [CS336 GitHub](https://github.com/stanford-cs336)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)
- [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [Dolma](https://allenai.org/dolma)
- [Chinchilla](https://arxiv.org/abs/2203.15556)

---

# Week 6 — Data engineering

## STUDY FIRST

Learn:

- [ ] web corpora
- [ ] document extraction
- [ ] quality filtering
- [ ] language detection
- [ ] PII filtering
- [ ] spam filtering
- [ ] deduplication
- [ ] near-duplicate detection
- [ ] contamination

Resources:

- [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [Dolma](https://allenai.org/dolma)
- [The Pile](https://pile.eleuther.ai/)
- [RedPajama](https://www.together.ai/blog/redpajama)

## IMPLEMENT

Build:

```text
06-pretraining/data_pipeline/
├── ingest.py
├── clean.py
├── deduplicate.py
├── quality.py
├── tokenize.py
├── shard.py
└── README.md
```

## EXPERIMENT

- [ ] Raw vs filtered.
- [ ] Dedup vs no dedup.
- [ ] Different quality thresholds.
- [ ] Measure dataset size after each stage.

## DOCUMENT

Produce:

```text
raw documents
→ cleaned
→ filtered
→ deduplicated
→ tokenized
→ shards
```

with counts at every stage.

---

# Week 7 — Pretraining theory

## STUDY

- [ ] CS336 training material.
- [ ] [Scaling Laws](https://arxiv.org/abs/2001.08361)
- [ ] [Chinchilla](https://arxiv.org/abs/2203.15556)

Learn:

- [ ] next-token objective
- [ ] cross entropy
- [ ] perplexity
- [ ] compute budget
- [ ] FLOPs
- [ ] tokens
- [ ] batch size
- [ ] optimizer states
- [ ] checkpointing
- [ ] scaling laws

## IMPLEMENT

- [ ] Pretraining loop for your Transformer.
- [ ] token counter.
- [ ] tokens/sec.
- [ ] estimated FLOPs.
- [ ] checkpoint recovery.

---

# Week 8–9 — MAJOR PROJECT #2: Pretrain your own model

Target:

```text
~50M–150M parameters
```

Your 3090 is intended for this kind of experiment.

- [ ] Build dataset.
- [ ] Build tokenizer.
- [ ] Build model.
- [ ] Train.
- [ ] Validate.
- [ ] checkpoint.
- [ ] generate samples.
- [ ] calculate perplexity.
- [ ] profile VRAM.
- [ ] record throughput.

## Publish

- [ ] Model card.
- [ ] Dataset card.
- [ ] Training configuration.
- [ ] Loss curve.
- [ ] Evaluation.
- [ ] Failure analysis.

Use Hugging Face Hub for the model where licensing permits:

- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index)
- [Model Cards](https://huggingface.co/docs/hub/model-cards)

---

# Week 10 — Pretraining ablations

Run controlled experiments:

- [ ] model size
- [ ] dataset size
- [ ] learning rate
- [ ] batch size
- [ ] sequence length
- [ ] tokenizer vocabulary

For every experiment record:

```text
hypothesis
configuration
result
interpretation
next experiment
```

---

# PHASE 5 — Modern Transformer architectures

# Weeks 11–13

## Week 11 — Modern decoder architecture

## STUDY

- [Llama 2](https://arxiv.org/abs/2307.09288)
- [Llama 3](https://arxiv.org/abs/2407.21783)
- [RoPE](https://arxiv.org/abs/2104.09864)

Learn:

- [ ] RMSNorm
- [ ] Pre-Norm
- [ ] SwiGLU
- [ ] RoPE
- [ ] MHA
- [ ] MQA
- [ ] GQA
- [ ] KV cache

## IMPLEMENT

Modify your Transformer:

- [ ] RMSNorm.
- [ ] SwiGLU.
- [ ] RoPE.
- [ ] GQA.

## EXPERIMENT

Compare:

```text
MHA vs GQA
absolute position vs RoPE
LayerNorm vs RMSNorm
```

Record VRAM, throughput and loss.

---

# Week 12 — Long context

## STUDY

- [ ] RoPE scaling
- [ ] sliding window attention
- [ ] KV cache
- [ ] context length
- [ ] long-context evaluation

## IMPLEMENT

- [ ] Increase your model context.
- [ ] Measure KV-cache memory.
- [ ] Measure generation speed.

---

# Week 13 — MoE

## STUDY

- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [Mixtral](https://arxiv.org/abs/2401.04088)
- [Megatron MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)

Learn:

- [ ] experts
- [ ] router
- [ ] top-k
- [ ] load balancing
- [ ] expert capacity
- [ ] active parameters
- [ ] total parameters

## IMPLEMENT — MAJOR PROJECT #3

Build a tiny MoE layer.

Compare dense vs MoE under similar compute.

---

# PHASE 6 — SFT

# Weeks 14–15

## Study first

Primary:

- [Hugging Face LLM Course — Fine-tuning](https://huggingface.co/learn/llm-course/en/chapter3/1)
- [Transformers Fine-tuning](https://huggingface.co/docs/transformers/main/training)
- [TRL](https://huggingface.co/docs/trl/)
- [Smol Course](https://huggingface.co/learn/smol-course/)

Learn:

- [ ] instruction dataset
- [ ] chat template
- [ ] supervised loss
- [ ] loss masking
- [ ] packing
- [ ] truncation
- [ ] catastrophic forgetting

## IMPLEMENT

Start with a small model.

- [ ] Build SFT dataset.
- [ ] Inspect tokenized samples.
- [ ] Inspect labels.
- [ ] Verify masked positions.
- [ ] Run a tiny overfit test.
- [ ] Train actual SFT.

## EXPERIMENT

- [ ] Different learning rates.
- [ ] sequence lengths.
- [ ] LoRA vs full/partial training.

## DOCUMENT

Your most important question:

> What exactly receives the loss during conversational SFT?

---

# PHASE 7 — LoRA / QLoRA

# Weeks 16–17

## STUDY

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PEFT documentation](https://huggingface.co/docs/peft/)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/)

Learn:

- [ ] low-rank adaptation
- [ ] rank
- [ ] alpha
- [ ] target modules
- [ ] adapter weights
- [ ] quantization
- [ ] NF4
- [ ] double quantization

## IMPLEMENT — MAJOR PROJECT #4

Fine-tune a 1B–3B model.

Compare:

```text
baseline
LoRA
QLoRA
```

Record:

- [ ] trainable parameters
- [ ] VRAM
- [ ] training time
- [ ] checkpoint size
- [ ] quality

---

# PHASE 8 — Evaluation

# Week 18

## STUDY

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/)
- [OpenAI Evals](https://github.com/openai/evals)

Learn:

- [ ] benchmark design
- [ ] exact match
- [ ] F1
- [ ] perplexity
- [ ] MMLU
- [ ] GSM8K
- [ ] GPQA
- [ ] HumanEval
- [ ] IFEval
- [ ] LongBench
- [ ] LLM-as-judge
- [ ] judge bias
- [ ] contamination

## IMPLEMENT — MAJOR PROJECT #5

Build:

```text
evaluation/
├── datasets/
├── runners/
├── metrics/
├── judges/
├── reports/
└── README.md
```

Evaluate your base/SFT/DPO models with the same protocol.

---

# PHASE 9 — RLHF + preference learning

# Weeks 19–21

## Week 19 — RLHF

## STUDY

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [TRL documentation](https://huggingface.co/docs/trl/)

Understand:

```text
base
 ↓
SFT
 ↓
preference data
 ↓
reward model
 ↓
PPO
 ↓
aligned model
```

Learn:

- [ ] reward model
- [ ] Bradley-Terry
- [ ] policy
- [ ] value
- [ ] advantage
- [ ] PPO
- [ ] KL penalty

---

# Week 20 — DPO

## STUDY

- [DPO paper](https://arxiv.org/abs/2305.18290)
- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL Quickstart](https://huggingface.co/docs/trl/quickstart)

Current TRL provides dedicated SFT, DPO, GRPO, reward-model and other post-training trainers. citeturn0search0turn0search1

Learn:

- [ ] chosen response
- [ ] rejected response
- [ ] reference model
- [ ] beta
- [ ] preference probability
- [ ] DPO loss

## IMPLEMENT

First:

- [ ] Implement the DPO loss yourself in PyTorch.
- [ ] Test it on synthetic preference examples.
- [ ] Verify gradients.
- [ ] Then use `DPOTrainer`.

## EXPERIMENT

Compare:

```text
SFT
DPO
```

under the same evaluation suite.

---

# Week 21 — Preference optimization family

## STUDY

- [ORPO](https://arxiv.org/abs/2403.07691)
- [KTO documentation](https://huggingface.co/docs/trl/kto_trainer)
- [TRL trainer overview](https://huggingface.co/docs/trl/)

Learn:

- [ ] IPO
- [ ] KTO
- [ ] ORPO
- [ ] SimPO
- [ ] online vs offline preference optimization

## EXPERIMENT

Train at least two methods.

Document:

> When does each method make sense?

---

# PHASE 10 — RL, GRPO and reasoning

# Weeks 22–25

## Week 22 — RL fundamentals

## STUDY

- [Spinning Up](https://spinningup.openai.com/)
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/)

Learn:

- [ ] MDP
- [ ] return
- [ ] reward
- [ ] policy
- [ ] value
- [ ] advantage
- [ ] policy gradient
- [ ] REINFORCE
- [ ] actor-critic
- [ ] PPO

## IMPLEMENT

Implement REINFORCE on a simple environment before applying RL to an LLM.

---

# Week 23 — Reward modeling

- [ ] Build preference pairs.
- [ ] Train a tiny reward model.
- [ ] Validate reward ranking.
- [ ] Test reward hacking.

Document:

```text
prompt
response A
response B
human/preference label
reward A
reward B
```

---

# Week 24 — GRPO

## STUDY

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [TRL examples](https://huggingface.co/docs/trl/en/example_overview)

Current TRL examples include self-contained SFT/GRPO examples with scripts, prompts, chat templates and evaluation code. citeturn0search5

Learn:

- [ ] group sampling
- [ ] relative reward
- [ ] advantage
- [ ] verifier
- [ ] KL control

## IMPLEMENT

- [ ] Create N responses per prompt.
- [ ] Build a deterministic verifier.
- [ ] Calculate rewards.
- [ ] Implement a tiny GRPO update.
- [ ] Then reproduce with TRL.

---

# Week 25 — MAJOR PROJECT #6: Tiny reasoning model

Use a task with automatically verifiable answers.

```text
problem
 ↓
N generations
 ↓
verifier
 ↓
reward
 ↓
GRPO
 ↓
new policy
```

Measure:

- [ ] base accuracy
- [ ] SFT accuracy
- [ ] DPO accuracy
- [ ] GRPO/RLVR accuracy
- [ ] reward
- [ ] response length

Publish a serious technical report.

---

# PHASE 11 — Distillation + synthetic data

# Weeks 26–27

## STUDY

- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [Self-Instruct](https://arxiv.org/abs/2212.10560)
- [TRL distillation](https://huggingface.co/docs/trl/)

Learn:

- [ ] teacher/student
- [ ] logits distillation
- [ ] response distillation
- [ ] synthetic instruction data
- [ ] rejection sampling
- [ ] data filtering

## IMPLEMENT

```text
teacher
 ↓
generate
 ↓
filter
 ↓
student
 ↓
SFT/distillation
```

## DOCUMENT

Measure whether synthetic data actually improved the student.

---

# PHASE 12 — Inference engineering

# Weeks 28–30

## Week 28 — Model memory

## STUDY

- [Hugging Face Performance Guide](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
- [vLLM documentation](https://docs.vllm.ai/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Learn:

- [ ] weights
- [ ] activations
- [ ] optimizer states
- [ ] KV cache
- [ ] prefill
- [ ] decode

## IMPLEMENT

Write a script that estimates:

```text
weight memory
KV cache memory
activation memory
```

for different model sizes and context lengths.

---

# Week 29 — Quantization

## STUDY

- [GPTQ](https://github.com/IST-DASLab/gptq)
- [AWQ](https://github.com/mit-han-lab/llm-awq)
- [bitsandbytes](https://huggingface.co/docs/bitsandbytes/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Learn:

- [ ] INT8
- [ ] INT4
- [ ] GPTQ
- [ ] AWQ
- [ ] GGUF
- [ ] weight-only quantization

## EXPERIMENT

Quantize a model.

Measure:

- [ ] VRAM
- [ ] tokens/sec
- [ ] latency
- [ ] quality loss

---

# Week 30 — vLLM/SGLang

## STUDY

- [vLLM](https://docs.vllm.ai/)
- [SGLang](https://docs.sglang.ai/)
- [PagedAttention](https://arxiv.org/abs/2309.06180)

Learn:

- [ ] continuous batching
- [ ] PagedAttention
- [ ] prefix caching
- [ ] speculative decoding
- [ ] TTFT
- [ ] throughput

## IMPLEMENT — MAJOR PROJECT #7

Deploy a local OpenAI-compatible API.

Run load tests.

Publish a benchmark table.

---

# PHASE 13 — Distributed training

# Weeks 31–34

## Week 31 — DDP

## STUDY

- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

Learn:

- [ ] process
- [ ] rank
- [ ] world size
- [ ] NCCL
- [ ] all-reduce

## IMPLEMENT

Run a toy distributed training experiment.

If you only have one GPU, understand the mechanics and document the limitation.

---

# Week 32 — FSDP

## STUDY

- [FSDP documentation](https://pytorch.org/docs/stable/fsdp.html)

Learn:

- [ ] parameter sharding
- [ ] gradient sharding
- [ ] optimizer sharding
- [ ] activation checkpointing

---

# Week 33 — ZeRO

## STUDY

- [DeepSpeed](https://www.deepspeed.ai/)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)

Learn:

- [ ] ZeRO-1
- [ ] ZeRO-2
- [ ] ZeRO-3

---

# Week 34 — Megatron

## STUDY

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
- [Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)

Learn:

- [ ] tensor parallelism
- [ ] pipeline parallelism
- [ ] context parallelism
- [ ] expert parallelism

---

# PHASE 14 — CUDA + Triton

# Weeks 35–37

## Week 35 — CUDA

## STUDY

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

Learn:

- [ ] thread
- [ ] block
- [ ] warp
- [ ] shared memory
- [ ] global memory
- [ ] register
- [ ] occupancy

## IMPLEMENT

- [ ] vector addition kernel
- [ ] reduction
- [ ] simple matrix operation

---

# Week 36 — FlashAttention

## STUDY

- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)

Understand:

> FlashAttention does not approximate attention; it changes how attention is computed and moved through the GPU memory hierarchy.

## EXPERIMENT

Compare memory/time of naive vs optimized attention.

---

# Week 37 — Triton

## STUDY

- [Triton](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

Implement:

- [ ] vector add
- [ ] softmax
- [ ] matmul
- [ ] benchmark vs PyTorch

Document the performance difference.

---

# PHASE 15 — RAG engineering

# Weeks 38–39

## STUDY

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Qdrant](https://qdrant.tech/documentation/)
- [BEIR](https://github.com/beir-cellar/beir)

Learn:

- [ ] BM25
- [ ] dense retrieval
- [ ] hybrid retrieval
- [ ] embeddings
- [ ] cross-encoder
- [ ] reranking
- [ ] Recall@K
- [ ] MRR
- [ ] nDCG
- [ ] faithfulness

## IMPLEMENT — MAJOR PROJECT #8

Build:

```text
documents
 ↓
chunking
 ↓
BM25 + vector retrieval
 ↓
fusion
 ↓
reranker
 ↓
LLM
```

Measure retrieval separately from answer generation.

---

# PHASE 16 — Agents

# Weeks 40–42

## STUDY

- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

Learn:

- [ ] function calling
- [ ] tool schemas
- [ ] ReAct
- [ ] planning
- [ ] memory
- [ ] tool failure
- [ ] observability
- [ ] agent evaluation

## IMPLEMENT — MAJOR PROJECT #9

Build an agent with:

- [ ] planner
- [ ] tools
- [ ] execution
- [ ] observation
- [ ] retries
- [ ] trace logs
- [ ] evaluation set
- [ ] cost tracking
- [ ] latency tracking

---

# PHASE 17 — Multimodal

# Weeks 43–46

## Vision

### STUDY

- [ViT](https://arxiv.org/abs/2010.11929)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Hugging Face Vision-Language Models](https://huggingface.co/docs/transformers/tasks/image_text_to_text)

Learn:

- [ ] ViT
- [ ] image encoder
- [ ] projector
- [ ] visual tokens
- [ ] VLM SFT

### IMPLEMENT

Build or fine-tune a small VLM.

---

## Audio

### STUDY

- [Whisper](https://github.com/openai/whisper)
- [Hugging Face Audio](https://huggingface.co/docs/transformers/tasks/asr)

Learn:

- [ ] audio encoder
- [ ] speech tokens
- [ ] ASR
- [ ] audio-language modeling

---

## Video

- [ ] frame sampling
- [ ] temporal representation
- [ ] video tokens
- [ ] multimodal context

---

# PHASE 18 — Interpretability

# Weeks 47–48

## STUDY

- [Transformer Circuits](https://transformer-circuits.pub/)
- [Neel Nanda](https://www.neelnanda.io/)
- [ARENA](https://www.arena.education/)

Learn:

- [ ] activation analysis
- [ ] probing
- [ ] causal tracing
- [ ] activation patching
- [ ] sparse autoencoders
- [ ] mechanistic interpretability

## IMPLEMENT

Pick one small Transformer and perform an interpretability experiment.

Document the hypothesis and evidence.

---

# PHASE 19 — Safety & alignment

# Week 49

## STUDY

- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Anthropic Research](https://www.anthropic.com/research)

Learn:

- [ ] hallucination
- [ ] sycophancy
- [ ] jailbreaks
- [ ] reward hacking
- [ ] specification gaming
- [ ] red teaming
- [ ] RLHF
- [ ] RLAIF

## IMPLEMENT

Create a small adversarial evaluation suite for one of your models.

---

# PHASE 20 — Research mode

# Week 50 onward

At this stage, courses become secondary.

Your workflow becomes:

```text
PAPER
 ↓
QUESTION
 ↓
DERIVE
 ↓
IMPLEMENT
 ↓
REPRODUCE
 ↓
BENCHMARK
 ↓
FIND DISCREPANCY
 ↓
HYPOTHESIS
 ↓
EXPERIMENT
 ↓
REPORT
```

## Paper sources

- [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent)
- [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent)
- [arXiv cs.AI](https://arxiv.org/list/cs.AI/recent)
- [Hugging Face Papers](https://huggingface.co/papers)
- [OpenReview](https://openreview.net/)

## Core paper sequence

- [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ ] [GPT-3](https://arxiv.org/abs/2005.14165)
- [ ] [Scaling Laws](https://arxiv.org/abs/2001.08361)
- [ ] [Chinchilla](https://arxiv.org/abs/2203.15556)
- [ ] [Llama 2](https://arxiv.org/abs/2307.09288)
- [ ] [InstructGPT](https://arxiv.org/abs/2203.02155)
- [ ] [LoRA](https://arxiv.org/abs/2106.09685)
- [ ] [QLoRA](https://arxiv.org/abs/2305.14314)
- [ ] [DPO](https://arxiv.org/abs/2305.18290)
- [ ] [FlashAttention](https://arxiv.org/abs/2205.14135)
- [ ] [ZeRO](https://arxiv.org/abs/1910.02054)
- [ ] [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [ ] [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [ ] [ORPO](https://arxiv.org/abs/2403.07691)
- [ ] [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

---

# 10. The GitHub portfolio architecture

Create a separate public repository:

```text
basim-llm-ml-mastery/
```

Recommended top-level README:

```markdown
# LLM & ML Mastery Journey

A public engineering journey from ML fundamentals to modern foundation-model
training, post-training, reasoning, inference and research.

## Hardware

- RTX 3090 24GB
- Intel i5-12600F
- 64GB RAM

## Objective

I am systematically studying and implementing the technologies behind
modern foundation models.

## Progress

- [ ] ML foundations
- [ ] NLP
- [ ] Attention
- [ ] Transformer from scratch
- [ ] Tokenization
- [ ] Pretraining
- [ ] Modern architectures
- [ ] SFT
- [ ] LoRA/QLoRA
- [ ] DPO
- [ ] RLHF
- [ ] GRPO/RLVR
- [ ] Distillation
- [ ] Evaluation
- [ ] Inference
- [ ] Distributed training
- [ ] CUDA/Triton
- [ ] RAG
- [ ] Agents
- [ ] Multimodal
- [ ] Interpretability
- [ ] Safety
- [ ] Research

## Major Projects

| Project | Status | Evidence |
|---|---|---|
| GPT from scratch | | |
| Custom tokenizer | | |
| Foundation model pretraining | | |
| SFT | | |
| LoRA/QLoRA | | |
| DPO | | |
| GRPO/RLVR | | |
| Evaluation framework | | |
| Inference server | | |
| Distributed training | | |
| RAG | | |
| Agent | | |
| Multimodal | | |
| Capstone | | |
```

---

# 11. Make every module recruiter/researcher readable

Every project README should answer these questions:

## 1. Problem

What were you trying to solve?

## 2. Theory

What concepts are involved?

## 3. Implementation

What did you actually code?

## 4. Hardware

What hardware did you use?

## 5. Experiment

What hypothesis did you test?

## 6. Results

What happened?

## 7. Failure

What broke?

## 8. Debugging

How did you identify the problem?

## 9. Fix

What changed?

## 10. Lessons

What did you learn?

## 11. Limitations

What does your experiment NOT prove?

## 12. Next experiment

What would you test next?

This format is much stronger for graduate applications and technical hiring than simply listing technologies.

---

# 12. Your experiment tracking standard

For every meaningful training experiment save:

```yaml
experiment:
  name:
  date:
  git_commit:

hardware:
  gpu:
  vram:
  cpu:
  ram:

software:
  python:
  pytorch:
  cuda:
  transformers:
  trl:

model:
  name:
  parameters:
  context_length:

data:
  dataset:
  examples:
  tokens:

training:
  epochs:
  batch_size:
  gradient_accumulation:
  learning_rate:
  scheduler:
  optimizer:
  precision:

results:
  train_loss:
  eval_loss:
  perplexity:
  tokens_per_second:
  max_vram:

notes:
  hypothesis:
  outcome:
  failure:
  fix:
  next_step:
```

Commit this alongside the code.

---

# 13. Experiment naming

Use:

```text
EXP-001-bigram-baseline
EXP-002-rnn-baseline
EXP-003-lstm-baseline
EXP-004-attention-baseline
EXP-005-gpt-small
EXP-006-gpt-large
EXP-007-rope
EXP-008-gqa
EXP-009-sft
EXP-010-lora
EXP-011-qlora
EXP-012-dpo
EXP-013-orpo
EXP-014-grpo
...
```

This creates a chronological research record.

---

# 14. Debugging journal

Create:

```text
DEBUGGING_LOG.md
```

Use:

```markdown
# Bug: Loss suddenly became NaN

## Symptoms

Training loss became NaN after step 4,200.

## Hypotheses

1. Learning rate too high.
2. Bad input.
3. Numerical instability.
4. Gradient explosion.

## Investigation

- Checked input → valid.
- Checked gradients → huge.
- Checked LR → normal.
- Checked mixed precision → issue reproduced.

## Root cause

...

## Fix

...

## Validation

...

## Lesson

...
```

This is extremely valuable portfolio material because **real engineering is mostly problem diagnosis**.

---

# 15. How to turn technical work into LinkedIn content

Never write:

> "Today I learned DPO."

Instead:

> I implemented DPO from the objective rather than starting with `DPOTrainer`.

Then explain:

```text
Problem
↓
Preference pairs
↓
Reference policy
↓
DPO objective
↓
Implementation
↓
Experiment
↓
Result
```

Then link:

- GitHub implementation
- experiment report
- model/dataset if public

Your posts should demonstrate:

**knowledge + implementation + evidence + reflection.**

---

# 16. Graduate-school portfolio strategy

For an M.S. application, preserve evidence of:

### Technical depth

- [ ] mathematical derivations
- [ ] implementation
- [ ] experiments
- [ ] ablations

### Research ability

- [ ] paper reproduction
- [ ] hypothesis
- [ ] experiment design
- [ ] analysis
- [ ] limitations

### Engineering ability

- [ ] clean repository
- [ ] tests
- [ ] reproducible environments
- [ ] configuration management
- [ ] benchmarks

### Communication

- [ ] technical reports
- [ ] diagrams
- [ ] concise README files
- [ ] research-style conclusions

### Open-source contribution

Eventually:

- [ ] fix an issue in an ML project.
- [ ] submit a documentation PR.
- [ ] submit a bug fix.
- [ ] contribute an example.
- [ ] contribute a benchmark.

That is stronger evidence than another generic certificate.

---

# 17. Your portfolio progression

Aim for this progression:

```text
LEVEL 1
"I understand PyTorch."

        ↓

LEVEL 2
"I implemented attention."

        ↓

LEVEL 3
"I implemented GPT from scratch."

        ↓

LEVEL 4
"I pretrained my own model."

        ↓

LEVEL 5
"I instruction-tuned a model."

        ↓

LEVEL 6
"I implemented preference optimization."

        ↓

LEVEL 7
"I trained a reasoning model with verifiable rewards."

        ↓

LEVEL 8
"I optimized inference."

        ↓

LEVEL 9
"I understand distributed training."

        ↓

LEVEL 10
"I reproduced research."

        ↓

LEVEL 11
"I designed my own experiment."

        ↓

LEVEL 12
"I have an independent research result."
```

The last four levels are where your profile becomes particularly compelling.

---

# 18. Definition of COMPLETE

Do not check a topic until:

- [ ] I studied the recommended primary material.
- [ ] I can explain the concept without notes.
- [ ] I understand why it exists.
- [ ] I understand the important mathematics.
- [ ] I implemented a minimal version.
- [ ] I used the production implementation.
- [ ] I ran at least one controlled experiment.
- [ ] I measured the result.
- [ ] I documented a failure or limitation.
- [ ] I committed the implementation.
- [ ] I published the learning log.
- [ ] I can explain it in an interview.

---

# 19. First 7 days — do this now

## Day 1

- [ ] Read PyTorch Get Started.
- [ ] Install/verify environment.
- [ ] Run GPU test.
- [ ] Create GitHub repository.
- [ ] Create `00-environment`.
- [ ] Create `LEARNING_LOG.md`.

## Day 2

- [ ] Read Dataset/DataLoader tutorial.
- [ ] Implement Dataset.
- [ ] Implement DataLoader.
- [ ] Benchmark batch sizes.
- [ ] Document.

## Day 3

- [ ] Study collate functions.
- [ ] Implement custom collate.
- [ ] Test workers.
- [ ] Benchmark.
- [ ] Document.

## Day 4

- [ ] Study autograd.
- [ ] Implement training loop.
- [ ] Add validation.
- [ ] Add checkpointing.
- [ ] Test resume.

## Day 5

- [ ] Study AdamW.
- [ ] Implement scheduler.
- [ ] Implement gradient accumulation.
- [ ] Run controlled experiment.

## Day 6

- [ ] Study AMP.
- [ ] Compare FP32/FP16/BF16.
- [ ] Measure VRAM.
- [ ] Measure throughput.

## Day 7

- [ ] Rebuild training loop without notes.
- [ ] Review failures.
- [ ] Write weekly report.
- [ ] Commit everything.
- [ ] Push GitHub.
- [ ] Publish a short LinkedIn progress post if the work is substantive.

---

# 20. Your long-term evidence tree

By the end, your public repository should look approximately like:

```text
basim-llm-ml-mastery/
│
├── README.md
├── ROADMAP.md
├── PROGRESS.md
├── DEBUGGING_LOG.md
│
├── weekly/
│   ├── 2026-W41.md
│   ├── 2026-W42.md
│   └── ...
│
├── 00-environment/
├── 01-ml-foundations/
├── 02-nlp-foundations/
├── 03-attention/
├── 04-transformer-from-scratch/
├── 05-tokenization/
├── 06-pretraining/
├── 07-modern-transformers/
├── 08-sft/
├── 09-lora-qlora/
├── 10-evaluation/
├── 11-preference-learning/
├── 12-rlhf/
├── 13-grpo-rlvr/
├── 14-distillation/
├── 15-inference/
├── 16-distributed-training/
├── 17-cuda-triton/
├── 18-rag/
├── 19-agents/
├── 20-multimodal/
├── 21-interpretability/
├── 22-safety-alignment/
│
└── 23-capstone/
```

---

# 21. The operating principle

The objective is **not**:

```text
finish 23 phases
```

The objective is:

```text
learn
→ build
→ break
→ debug
→ measure
→ explain
→ publish
→ repeat
```

If a concept takes two weeks because you discovered a difficult implementation problem, that is **not falling behind**.

That is the training.

---

# 22. Final capstone

After completing the curriculum, build:

```text
RAW DATA
   ↓
DATA ENGINEERING
   ↓
TOKENIZER
   ↓
PRETRAINING
   ↓
BASE MODEL
   ↓
SFT
   ↓
PREFERENCE DATA
   ↓
DPO
   ↓
RLVR / GRPO
   ↓
EVALUATION
   ↓
QUANTIZATION
   ↓
INFERENCE SERVER
   ↓
RAG
   ↓
TOOLS
   ↓
AGENT
   ↓
TECHNICAL REPORT
```

Your final repository should contain:

- [ ] source code
- [ ] configs
- [ ] experiment logs
- [ ] results
- [ ] figures
- [ ] model card
- [ ] dataset card
- [ ] benchmark report
- [ ] debugging history
- [ ] research report
- [ ] reproducibility instructions

---

# 23. What success looks like

At the end of this journey, you should be able to take a model such as a modern Qwen/Llama/Mistral/DeepSeek-family model and explain:

```text
Where did the training data come from?
        ↓
How was it filtered?
        ↓
How was it tokenized?
        ↓
Why this architecture?
        ↓
How does attention work?
        ↓
How was it pretrained?
        ↓
Why this learning-rate schedule?
        ↓
How was it instruction tuned?
        ↓
Why SFT?
        ↓
Why DPO?
        ↓
Why/when RL?
        ↓
How does GRPO work?
        ↓
How is reasoning rewarded?
        ↓
How is the model evaluated?
        ↓
How much memory does inference require?
        ↓
How can it be quantized?
        ↓
How can it be served efficiently?
        ↓
How does distributed training work?
        ↓
How can the model be connected to tools?
        ↓
How can its behavior be investigated?
        ↓
What experiment would improve it?
```

That is the target competency.

---

# Recommended primary resource stack

Use these as your backbone rather than collecting hundreds of random tutorials:

1. [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/)
2. [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)
3. [Hugging Face Smol Course](https://huggingface.co/learn/smol-course/)
4. [Hugging Face TRL](https://huggingface.co/docs/trl/)
5. [PyTorch](https://pytorch.org/)
6. [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)
7. [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
8. [vLLM](https://docs.vllm.ai/)
9. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
10. [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/)
11. [Triton](https://triton-lang.org/)
12. [FlashAttention](https://github.com/Dao-AILab/flash-attention)
13. [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent)
14. [arXiv cs.LG](https://arxiv.org/list/cs.LG/recent)
15. [Hugging Face Papers](https://huggingface.co/papers)

---

# Final rule

**Never write "I learned X" when you can write "I implemented X, measured Y, encountered Z, fixed it by doing A, and the result changed from B to C."**

That is the difference between a course completion record and an engineering/research portfolio.
